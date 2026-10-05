from pathlib import Path

from data_compress.ephys.probe import build_probe_report
from tests.ephys.spikeglx_fixtures import make_traces, write_spikeglx


def test_probe_report_summarizes_recording(tmp_path: Path) -> None:
    bin_path = write_spikeglx(tmp_path, make_traces(60000, n_neural=8))

    report = build_probe_report(bin_path)

    assert report.stream == "ap"
    assert report.n_neural_chans == 8
    assert report.n_sync_chans == 1
    assert report.sample_rate == 30000.0
    assert report.n_samples == 60000
    assert report.duration_s == 2.0
    assert report.size_bytes == 60000 * 9 * 2
    assert not report.size_mismatch


def test_probe_report_flags_size_mismatch_instead_of_failing(tmp_path: Path) -> None:
    bin_path = write_spikeglx(tmp_path, make_traces(100, n_neural=8), file_size_bytes=12345)

    report = build_probe_report(bin_path)

    assert report.size_mismatch
