
import os


def run():
    assert os.environ.get('SREGNAS_X25_UNTIMED')=='1','default-off untimed entry requires explicit activation'
    from sregnas.icde27 import recorder
    original=recorder.run_recorded
    def quality_record(**kwargs):
        assert kwargs.get('quality_only') is True,'untimed entry cannot change a timing run'
        kwargs['timeout']=None
        return original(**kwargs)
    recorder.run_recorded=quality_record
    from sregnas.comparison.launch import main
    return main()


if __name__=='__main__':raise SystemExit(run())
