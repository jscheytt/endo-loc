import subprocess
import os

import pytest

import helper.helper as hlp
import tests.conftest as cft


def run_python_subprocess(*args):
    return subprocess.run(["python", *args], cwd=pytest.config.rootdir.strpath)


def test_convert_labels():
    length_orig = hlp.file_length(cft.eval_label_list)
    ret = run_python_subprocess("convert_labels_to_label_list.py", cft.eval_labels, cft.eval_video_ft)
    assert ret.returncode == 0
    assert ret.stderr is None
    assert ret.stdout is None
    assert hlp.file_length(cft.eval_label_list) == length_orig


@pytest.mark.skip(reason="Intense")
def test_convert_video():
    import convert_video_to_xml as script
    permutations = [x for x in hlp.get_binary_bool_permutations(3) if True in x]
    for bool_list in permutations:
        h, s, v = bool_list
        no_h, no_s, no_v = [not x for x in bool_list]
        h_c = s_c = v_c = ''
        if no_h:
            h_c = "-c"
        if no_s:
            s_c = "-s"
        if no_v:
            v_c = "-v"
        full_args = [h_c, s_c, v_c]
        args = [x for x in full_args if x is not '']
        target_filename = script.get_default_target_filename(cft.exp_video, h, s, v)
        while True:
            # TODO Actually this is supposed to keep running until the file is created.
            # However the test still terminates if not all of the files exist.
            length_orig = hlp.file_length(target_filename)
            ret = run_python_subprocess("convert_video_to_xml.py", cft.exp_video, *args)
            if os.path.isfile(target_filename) or length_orig != -1:
                break
        assert ret.returncode == 0
        assert ret.stderr is None
        assert ret.stdout is None
        assert hlp.file_length(target_filename) == length_orig


