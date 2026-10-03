"""
test_automation.py
Tests for the automation engine: scheduling, queueing, workflow
definition/execution, and trigger firing.
"""

from __future__ import annotations

import time

import pytest

from conftest import MockAutomationEngine


def test_schedule_creates_job_with_unique_id(mock_automation: MockAutomationEngine) -> None:
    job = mock_automation.schedule("nightly_backup", run_at=time.time() + 3600)
    assert job.id in mock_automation.scheduler
    assert job.triggered is False


def test_enqueue_and_dequeue_preserve_fifo_order(mock_automation: MockAutomationEngine) -> None:
    mock_automation.enqueue({"id": "1"})
    mock_automation.enqueue({"id": "2"})

    first = mock_automation.dequeue()
    second = mock_automation.dequeue()

    assert first == {"id": "1"}
    assert second == {"id": "2"}


def test_dequeue_on_empty_queue_returns_none(mock_automation: MockAutomationEngine) -> None:
    assert mock_automation.dequeue() is None


def test_define_workflow_stores_step_sequence(mock_automation: MockAutomationEngine) -> None:
    mock_automation.define_workflow("onboarding", ["create_account", "send_welcome_email"])
    assert mock_automation.workflows["onboarding"] == ["create_account", "send_welcome_email"]


@pytest.mark.asyncio
async def test_run_workflow_executes_steps_in_order(mock_automation: MockAutomationEngine) -> None:
    mock_automation.define_workflow("deploy", ["build", "test", "release"])
    executed = await mock_automation.run_workflow("deploy")
    assert executed == ["build", "test", "release"]


@pytest.mark.asyncio
async def test_run_undefined_workflow_returns_empty_list(mock_automation: MockAutomationEngine) -> None:
    executed = await mock_automation.run_workflow("does_not_exist")
    assert executed == []


def test_fire_trigger_marks_job_triggered_and_logs_it(mock_automation: MockAutomationEngine) -> None:
    job = mock_automation.schedule("report_generation", run_at=time.time())
    fired = mock_automation.fire_trigger(job.id)

    assert fired is True
    assert mock_automation.scheduler[job.id].triggered is True
    assert job.id in mock_automation.trigger_log


def test_fire_trigger_for_unknown_job_returns_false(mock_automation: MockAutomationEngine) -> None:
    assert mock_automation.fire_trigger("missing-job") is False


@pytest.mark.asyncio
async def test_scheduled_job_can_drive_a_queued_workflow_run(mock_automation: MockAutomationEngine) -> None:
    job = mock_automation.schedule("weekly_digest", run_at=time.time())
    mock_automation.define_workflow("weekly_digest", ["aggregate", "summarize", "send"])

    mock_automation.fire_trigger(job.id)
    mock_automation.enqueue({"job_id": job.id, "workflow": "weekly_digest"})

    task = mock_automation.dequeue()
    assert task is not None
    executed = await mock_automation.run_workflow(task["workflow"])

    assert executed == ["aggregate", "summarize", "send"]
    assert mock_automation.scheduler[job.id].triggered is True