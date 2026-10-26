
from lerobot.utils.manual_recording import FileKeyboard, apply_key, manual_episodes


def events():
    return {"exit_early": False, "rerecord_episode": False, "stop_recording": False}


class Keyboard:
    def __init__(self, e):
        self.events = e
        self.phase = None

    def set_phase(self, p):
        self.phase = p
        self.events.update(exit_early=False, rerecord_episode=False)

    def poll(self):
        pass


class Dataset:
    def __init__(self):
        self.num_episodes = 0
        self.episode_buffer = {"size": 0}
        self.saved = []
        self.discarded = 0

    def clear_episode_buffer(self):
        self.episode_buffer = {"size": 0}
        self.discarded += 1

    def save_episode(self):
        self.saved.append(self.episode_buffer["size"])
        self.num_episodes += 1
        self.episode_buffer = {"size": 0}


def test_two_explicit_starts_and_finishes_no_wait_frames_recorded():
    e = events()
    k = Keyboard(e)
    d = Dataset()
    phases = []

    def run(dataset, poll):
        phases.append(k.phase)
        if dataset is None:
            apply_key(e, k.phase, "s")
        else:
            dataset.episode_buffer["size"] = 20
            apply_key(e, k.phase, "f")

    assert manual_episodes(d, e, k, run, 2, lambda _: None) == 2
    assert phases == ["waiting", "recording", "waiting", "recording"]
    assert d.saved == [20, 20]


def test_discard_does_not_increment_then_save_and_quit():
    e = events()
    k = Keyboard(e)
    d = Dataset()
    actions = iter(["d", "q"])

    def run(dataset, poll):
        if dataset is None:
            apply_key(e, k.phase, "s")
        else:
            dataset.episode_buffer["size"] = 12
            apply_key(e, k.phase, next(actions))

    assert manual_episodes(d, e, k, run, 25, lambda _: None) == 1
    assert d.discarded == 1 and d.saved == [12]


def test_quit_waiting_never_creates_empty_episode():
    e = events()
    d = Dataset()
    k = Keyboard(e)

    def run(dataset, poll):
        assert dataset is None
        apply_key(e, k.phase, "q")

    assert manual_episodes(d, e, k, run, 25, lambda _: None) == 0
    assert not d.saved


def test_five_minute_timeout_can_save_without_finish_key():
    e = events()
    d = Dataset()
    k = Keyboard(e)

    def run(dataset, poll):
        if dataset is None:
            apply_key(e, k.phase, "s")
        else:
            dataset.episode_buffer["size"] = 9000  # Timed control loop returned normally.

    assert manual_episodes(d, e, k, run, 1, lambda _: None) == 1
    assert d.saved == [9000]


def test_file_control_consumes_once_and_discard_exits_attempt(tmp_path):
    p = tmp_path / "commands"
    p.write_text("s\n")
    e = events()
    k = FileKeyboard(e, str(p)).start()
    try:
        k.set_phase("waiting")
        k.poll()
        assert e["exit_early"]
        k.set_phase("recording")
        k.poll()
        assert not e["exit_early"]
        with p.open("a") as f:
            f.write("d\n")
        k.poll()
        assert e["rerecord_episode"] and e["stop_recording"]
    finally:
        k.stop()
