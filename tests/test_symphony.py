"""Tests for Symphony orchestration service.

Covers: WorkflowLoader parsing, workspace key sanitization, config validation,
state machine transitions, retry backoff calculation, prompt rendering.
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# Add project root so we can import symphony
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import symphony


# ---------------------------------------------------------------------------
# WorkflowLoader
# ---------------------------------------------------------------------------
class TestWorkflowLoader(unittest.TestCase):
    """Parse WORKFLOW.md YAML frontmatter + body."""

    SAMPLE_WORKFLOW = textwrap.dedent("""\
        # Parameter Golf Autonomous Experiment Workflow

        ## Metadata

        ```yaml
        ---
        name: parameter-golf-experiment
        trigger: linear_issue
        issue_filter: "EXP-*"
        max_concurrent: 4
        timeout: 900
        ---
        ```

        ## Overview

        This is the prompt body with {{issue_title}} and {{issue_description}}.
    """)

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        )
        self.tmp.write(self.SAMPLE_WORKFLOW)
        self.tmp.close()
        self.addCleanup(os.unlink, self.tmp.name)

    def test_parses_yaml_frontmatter(self):
        wf = symphony.WorkflowLoader(self.tmp.name)
        cfg = wf.load()
        self.assertEqual(cfg["name"], "parameter-golf-experiment")
        self.assertEqual(cfg["trigger"], "linear_issue")
        self.assertEqual(cfg["issue_filter"], "EXP-*")
        self.assertEqual(cfg["max_concurrent"], 4)
        self.assertEqual(cfg["timeout"], 900)

    def test_extracts_prompt_body(self):
        wf = symphony.WorkflowLoader(self.tmp.name)
        wf.load()
        body = wf.prompt_body
        self.assertIn("{{issue_title}}", body)
        self.assertIn("{{issue_description}}", body)

    def test_reload_picks_up_changes(self):
        wf = symphony.WorkflowLoader(self.tmp.name)
        wf.load()
        self.assertEqual(wf.config["max_concurrent"], 4)
        # Rewrite with different value
        with open(self.tmp.name, "w") as f:
            f.write(self.SAMPLE_WORKFLOW.replace("max_concurrent: 4", "max_concurrent: 8"))
        cfg2 = wf.load()
        self.assertEqual(cfg2["max_concurrent"], 8)

    def test_missing_file_raises(self):
        wf = symphony.WorkflowLoader("/nonexistent/WORKFLOW.md")
        with self.assertRaises(FileNotFoundError):
            wf.load()

    def test_no_yaml_block_returns_defaults(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Just a heading\n\nSome body text.")
            f.flush()
            self.addCleanup(os.unlink, f.name)
            wf = symphony.WorkflowLoader(f.name)
            cfg = wf.load()
            # Should return empty/defaults without crashing
            self.assertIsInstance(cfg, dict)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class TestConfig(unittest.TestCase):
    """Config validation and env-var expansion."""

    def test_defaults(self):
        cfg = symphony.Config({})
        self.assertEqual(cfg.max_concurrent, 4)
        self.assertEqual(cfg.poll_interval, 30)
        self.assertEqual(cfg.timeout, 900)
        self.assertIsNotNone(cfg.workspace_root)

    def test_overrides_from_workflow(self):
        cfg = symphony.Config({
            "max_concurrent": 8,
            "timeout": 600,
            "poll_interval": 10,
        })
        self.assertEqual(cfg.max_concurrent, 8)
        self.assertEqual(cfg.timeout, 600)
        self.assertEqual(cfg.poll_interval, 10)

    def test_env_var_expansion(self):
        os.environ["_TEST_SYMPHONY_KEY"] = "secret123"
        self.addCleanup(os.environ.pop, "_TEST_SYMPHONY_KEY", None)
        cfg = symphony.Config({"linear_api_key": "$_TEST_SYMPHONY_KEY"})
        self.assertEqual(cfg.linear_api_key, "secret123")

    def test_env_var_missing_stays_literal(self):
        cfg = symphony.Config({"linear_api_key": "$NONEXISTENT_VAR_XYZ"})
        # Should not crash; stays as empty or literal
        self.assertIsInstance(cfg.linear_api_key, str)

    def test_issue_filter_default(self):
        cfg = symphony.Config({})
        self.assertEqual(cfg.issue_filter, "*")

    def test_issue_filter_from_workflow(self):
        cfg = symphony.Config({"issue_filter": "EXP-*"})
        self.assertEqual(cfg.issue_filter, "EXP-*")


# ---------------------------------------------------------------------------
# Workspace key sanitization
# ---------------------------------------------------------------------------
class TestWorkspaceKeySanitization(unittest.TestCase):
    """Only [A-Za-z0-9._-] allowed; others become _."""

    def test_clean_key_unchanged(self):
        self.assertEqual(symphony.sanitize_workspace_key("EXP-42"), "EXP-42")

    def test_spaces_replaced(self):
        self.assertEqual(
            symphony.sanitize_workspace_key("my issue name"),
            "my_issue_name",
        )

    def test_special_chars_replaced(self):
        self.assertEqual(
            symphony.sanitize_workspace_key("feat/add@thing#1"),
            "feat_add_thing_1",
        )

    def test_dots_and_underscores_preserved(self):
        self.assertEqual(
            symphony.sanitize_workspace_key("v1.0_release"),
            "v1.0_release",
        )

    def test_empty_string(self):
        self.assertEqual(symphony.sanitize_workspace_key(""), "")

    def test_unicode_replaced(self):
        self.assertEqual(
            symphony.sanitize_workspace_key("issue-\u00e9\u00e8"),
            "issue-__",
        )


# ---------------------------------------------------------------------------
# State machine transitions
# ---------------------------------------------------------------------------
class TestStateMachine(unittest.TestCase):
    """Issue state machine: Unclaimed -> Claimed -> Running -> terminal."""

    def test_valid_transitions(self):
        sm = symphony.IssueStateMachine()
        sm.transition("issue-1", symphony.IssueState.UNCLAIMED, symphony.IssueState.CLAIMED)
        self.assertEqual(sm.get_state("issue-1"), symphony.IssueState.CLAIMED)

    def test_claimed_to_running(self):
        sm = symphony.IssueStateMachine()
        sm.transition("i1", symphony.IssueState.UNCLAIMED, symphony.IssueState.CLAIMED)
        sm.transition("i1", symphony.IssueState.CLAIMED, symphony.IssueState.RUNNING)
        self.assertEqual(sm.get_state("i1"), symphony.IssueState.RUNNING)

    def test_running_to_succeeded(self):
        sm = symphony.IssueStateMachine()
        sm.transition("i1", symphony.IssueState.UNCLAIMED, symphony.IssueState.CLAIMED)
        sm.transition("i1", symphony.IssueState.CLAIMED, symphony.IssueState.RUNNING)
        sm.transition("i1", symphony.IssueState.RUNNING, symphony.IssueState.SUCCEEDED)
        self.assertEqual(sm.get_state("i1"), symphony.IssueState.SUCCEEDED)

    def test_running_to_failed(self):
        sm = symphony.IssueStateMachine()
        sm.transition("i1", symphony.IssueState.UNCLAIMED, symphony.IssueState.CLAIMED)
        sm.transition("i1", symphony.IssueState.CLAIMED, symphony.IssueState.RUNNING)
        sm.transition("i1", symphony.IssueState.RUNNING, symphony.IssueState.FAILED)
        self.assertEqual(sm.get_state("i1"), symphony.IssueState.FAILED)

    def test_invalid_transition_raises(self):
        sm = symphony.IssueStateMachine()
        with self.assertRaises(symphony.InvalidTransitionError):
            sm.transition("i1", symphony.IssueState.UNCLAIMED, symphony.IssueState.RUNNING)

    def test_unknown_issue_returns_unclaimed(self):
        sm = symphony.IssueStateMachine()
        self.assertEqual(sm.get_state("unknown"), symphony.IssueState.UNCLAIMED)

    def test_failed_to_retry_queued(self):
        sm = symphony.IssueStateMachine()
        sm.transition("i1", symphony.IssueState.UNCLAIMED, symphony.IssueState.CLAIMED)
        sm.transition("i1", symphony.IssueState.CLAIMED, symphony.IssueState.RUNNING)
        sm.transition("i1", symphony.IssueState.RUNNING, symphony.IssueState.FAILED)
        sm.transition("i1", symphony.IssueState.FAILED, symphony.IssueState.RETRY_QUEUED)
        self.assertEqual(sm.get_state("i1"), symphony.IssueState.RETRY_QUEUED)

    def test_retry_queued_to_claimed(self):
        sm = symphony.IssueStateMachine()
        sm.transition("i1", symphony.IssueState.UNCLAIMED, symphony.IssueState.CLAIMED)
        sm.transition("i1", symphony.IssueState.CLAIMED, symphony.IssueState.RUNNING)
        sm.transition("i1", symphony.IssueState.RUNNING, symphony.IssueState.FAILED)
        sm.transition("i1", symphony.IssueState.FAILED, symphony.IssueState.RETRY_QUEUED)
        sm.transition("i1", symphony.IssueState.RETRY_QUEUED, symphony.IssueState.CLAIMED)
        self.assertEqual(sm.get_state("i1"), symphony.IssueState.CLAIMED)


# ---------------------------------------------------------------------------
# Retry backoff
# ---------------------------------------------------------------------------
class TestRetryBackoff(unittest.TestCase):
    """Exponential backoff: delay = min(10000 * 2^(attempt-1), max_backoff_ms)."""

    def test_first_attempt(self):
        self.assertEqual(symphony.retry_backoff_ms(1), 10000)

    def test_second_attempt(self):
        self.assertEqual(symphony.retry_backoff_ms(2), 20000)

    def test_third_attempt(self):
        self.assertEqual(symphony.retry_backoff_ms(3), 40000)

    def test_capped_at_max(self):
        # Default max is 300000 (5 min)
        result = symphony.retry_backoff_ms(100)
        self.assertEqual(result, 300000)

    def test_custom_max(self):
        result = symphony.retry_backoff_ms(10, max_backoff_ms=50000)
        self.assertEqual(result, 50000)

    def test_zero_attempt_uses_minimum(self):
        result = symphony.retry_backoff_ms(0)
        self.assertGreater(result, 0)


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------
class TestPromptBuilder(unittest.TestCase):
    """Render prompt template with issue context."""

    def test_basic_substitution(self):
        template = "Fix: {{issue_title}}\n\nDetails: {{issue_description}}"
        issue = {
            "title": "Bug in training loop",
            "description": "Loss diverges after step 500",
            "labels": ["bug"],
            "priority": 1,
            "identifier": "EXP-42",
        }
        result = symphony.PromptBuilder.render(template, issue, "/tmp/ws", "exp-42")
        self.assertIn("Bug in training loop", result)
        self.assertIn("Loss diverges after step 500", result)

    def test_workspace_path_injected(self):
        template = "Work in: {{workspace_path}}"
        issue = {"title": "t", "description": "d", "labels": [], "priority": 0, "identifier": "X-1"}
        result = symphony.PromptBuilder.render(template, issue, "/tmp/ws", "b")
        self.assertIn("/tmp/ws", result)

    def test_branch_name_injected(self):
        template = "Branch: {{branch_name}}"
        issue = {"title": "t", "description": "d", "labels": [], "priority": 0, "identifier": "X-1"}
        result = symphony.PromptBuilder.render(template, issue, "/w", "my-branch")
        self.assertIn("my-branch", result)

    def test_labels_as_comma_string(self):
        template = "Labels: {{issue_labels}}"
        issue = {"title": "t", "description": "d", "labels": ["bug", "urgent"], "priority": 0, "identifier": "X-1"}
        result = symphony.PromptBuilder.render(template, issue, "/w", "b")
        self.assertIn("bug", result)
        self.assertIn("urgent", result)

    def test_missing_placeholder_left_intact(self):
        template = "{{unknown_var}} stays"
        issue = {"title": "t", "description": "d", "labels": [], "priority": 0, "identifier": "X-1"}
        result = symphony.PromptBuilder.render(template, issue, "/w", "b")
        self.assertIn("{{unknown_var}}", result)


# ---------------------------------------------------------------------------
# WorkspaceManager safety invariants
# ---------------------------------------------------------------------------
class TestWorkspaceManagerSafety(unittest.TestCase):
    """Workspace paths must be inside workspace root."""

    def test_path_inside_root_ok(self):
        root = "/tmp/symphony_ws"
        path = symphony.WorkspaceManager.compute_workspace_path(root, "EXP-42")
        self.assertTrue(path.startswith(root))

    def test_traversal_attack_blocked(self):
        root = "/tmp/symphony_ws"
        # A malicious key that tries to escape
        with self.assertRaises(symphony.WorkspaceSafetyError):
            symphony.WorkspaceManager.compute_workspace_path(root, "../../etc/passwd")

    def test_absolute_path_blocked(self):
        root = "/tmp/symphony_ws"
        with self.assertRaises(symphony.WorkspaceSafetyError):
            symphony.WorkspaceManager.compute_workspace_path(root, "/etc/passwd")


# ---------------------------------------------------------------------------
# IssueTrackerClient filter matching
# ---------------------------------------------------------------------------
class TestIssueFilter(unittest.TestCase):
    """Filter issues by prefix pattern like 'EXP-*'."""

    def test_exp_star_matches(self):
        self.assertTrue(symphony.matches_filter("EXP-42", "EXP-*"))

    def test_exp_star_no_match(self):
        self.assertFalse(symphony.matches_filter("BUG-1", "EXP-*"))

    def test_wildcard_matches_all(self):
        self.assertTrue(symphony.matches_filter("ANYTHING-99", "*"))

    def test_exact_match(self):
        self.assertTrue(symphony.matches_filter("EXP-1", "EXP-1"))


# ---------------------------------------------------------------------------
# Priority sorting
# ---------------------------------------------------------------------------
class TestPrioritySorting(unittest.TestCase):
    """Issues sorted by priority (lower = higher prio) then creation date."""

    def test_sorts_by_priority(self):
        issues = [
            {"identifier": "A", "priority": 3, "createdAt": "2024-01-01"},
            {"identifier": "B", "priority": 1, "createdAt": "2024-01-02"},
            {"identifier": "C", "priority": 2, "createdAt": "2024-01-01"},
        ]
        sorted_issues = symphony.sort_issues_by_priority(issues)
        ids = [i["identifier"] for i in sorted_issues]
        self.assertEqual(ids, ["B", "C", "A"])

    def test_tiebreak_by_date(self):
        issues = [
            {"identifier": "A", "priority": 1, "createdAt": "2024-06-01"},
            {"identifier": "B", "priority": 1, "createdAt": "2024-01-01"},
        ]
        sorted_issues = symphony.sort_issues_by_priority(issues)
        ids = [i["identifier"] for i in sorted_issues]
        self.assertEqual(ids, ["B", "A"])


if __name__ == "__main__":
    unittest.main()
