"""The monitor thread must survive faults: if it dies, the training lease stops
refreshing and the worker cancels a healthy job after 120 seconds.
"""

import time

from recog.service import Service


class Flaky:
    """Reports as running, but the first poll raises like a racing stat() would."""

    def __init__(self):
        self.calls = 0
        self.returncode = None

    def poll(self):
        self.calls += 1
        if self.calls == 1:
            raise OSError("模拟的进程查询失败")
        return None


def test_monitor_survives_a_failing_tick(tmp_path):
    service = Service(tmp_path, python=str(tmp_path / "python"))
    job = "a" * 32
    directory = tmp_path / "jobs" / job
    directory.mkdir(parents=True)
    lease = directory / "lease"
    lease.touch()
    process = Flaky()
    service.processes[job] = process
    try:
        before = lease.stat().st_mtime_ns
        time.sleep(2.5)
        assert service.monitor.is_alive(), "监控线程不能因为一次异常就退出"
        assert lease.stat().st_mtime_ns > before, "租约必须继续刷新"
        assert process.calls >= 2
    finally:
        service.processes.pop(job, None)
        service.close()
