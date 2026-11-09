import numpy as np
import pytest
import soundfile as sf

from birdclef import config as C
from birdclef.prepare import convert, convert_all


def make_raw(root):
    clip = root / "train_audio" / "sp1" / "a.ogg"
    scape = root / "train_soundscapes" / "s.ogg"
    clip.parent.mkdir(parents=True)
    scape.parent.mkdir(parents=True)
    t = np.arange(2 * 44100) / 44100
    tone = 0.5 * np.sin(2 * np.pi * 1000 * t)
    sf.write(clip, np.stack([tone, tone], 1), 44100)  # stereo, 44.1 kHz
    sf.write(scape, 0.1 * np.random.default_rng(0).standard_normal(3 * C.SR), C.SR)  # mono, already at C.SR
    return root


def test_convert_to_wav_16bit_mono_at_sr(tmp_path):
    raw, wav = make_raw(tmp_path / "raw"), tmp_path / "wav"
    assert convert_all(raw, wav, workers=2) == 2
    for name, seconds in (("train_audio/sp1/a.wav", 2), ("train_soundscapes/s.wav", 3)):
        info = sf.info(wav / name)
        assert (info.samplerate, info.channels, info.subtype) == (C.SR, 1, "PCM_16")
        assert abs(info.frames - seconds * C.SR) < C.SR // 100
    x, _ = sf.read(wav / "train_audio/sp1/a.wav")
    spectrum = np.abs(np.fft.rfft(x))
    assert abs(np.argmax(spectrum) * C.SR / len(x) - 1000) < 5  # the tone survives resampling
    assert abs(np.abs(x).max() - 0.5) < 0.05


def test_rerun_skips_converted_files(tmp_path):
    raw, wav = make_raw(tmp_path / "raw"), tmp_path / "wav"
    convert_all(raw, wav, workers=2)
    mtime = (wav / "train_soundscapes/s.wav").stat().st_mtime_ns
    assert convert_all(raw, wav, workers=2) == 0
    assert (wav / "train_soundscapes/s.wav").stat().st_mtime_ns == mtime
    assert not list(wav.rglob("*.part.wav"))


def test_delete_src_only_after_wav_is_complete(tmp_path):
    raw, wav = make_raw(tmp_path / "raw"), tmp_path / "wav"
    assert convert_all(raw, wav, workers=2, delete_src=True) == 2
    assert not list(raw.rglob("*.ogg"))
    assert len(list(wav.rglob("*.wav"))) == 2

    bad = raw / "train_audio" / "sp1" / "bad.ogg"
    bad.write_bytes(b"not audio")
    with pytest.raises(sf.LibsndfileError):
        convert(bad, wav / "train_audio" / "sp1" / "bad.wav", delete_src=True)
    assert bad.exists()
    assert not (wav / "train_audio" / "sp1" / "bad.wav").exists()
