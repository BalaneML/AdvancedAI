"""Author-equivalent cv2 extraction, with per-video resume and atomic PNG writes."""
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/JAAD'
def extract_video(video):
    cv2.setNumThreads(1)
    video_started = time.time()
    tree = ET.parse(DATA / 'annotations' / (video.stem + '.xml'))
    expected = int(tree.find('./meta/task/size').text)
    folder = DATA / 'images' / video.stem
    folder.mkdir(parents=True, exist_ok=True)
    expected_names = {f'{i:05d}.png' for i in range(expected)}
    actual_names = {p.name for p in folder.glob('*.png') if p.stem.isdigit()}
    if actual_names == expected_names:
        for i in {0, expected // 2, expected - 1}:
            assert cv2.imread(str(folder / f'{i:05d}.png')) is not None
        print(f'{video.stem}: skip complete {expected}', flush=True)
    else:
        # Bound FFmpeg decoder threads as well as OpenCV threads. Each worker
        # owns a separate video; no two workers write the same destination.
        cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_N_THREADS, 1])
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            path = folder / f'{count:05d}.png'
            if not path.exists() or cv2.imread(str(path)) is None:
                if path.exists():
                    path.rename(path.with_suffix(f'.corrupt_{time.time_ns()}'))
                temporary = folder / f'.{count:05d}.partial.png'
                assert cv2.imwrite(str(temporary), frame), temporary
                os.replace(temporary, path)
            count += 1
        cap.release()
        assert count == expected, f'{video.stem}: decoded {count}, XML {expected}'
        assert {p.name for p in folder.glob('*.png') if p.stem.isdigit()} == expected_names
        print(f'{video.stem}: exported {count}', flush=True)
    sample = cv2.imread(str(folder / '00000.png'))
    return {'video': video.stem, 'frames': expected, 'width': sample.shape[1],
            'height': sample.shape[0], 'seconds': time.time()-video_started,
            'png_bytes': sum(p.stat().st_size for p in folder.glob('*.png'))}

def main():
    started = time.time()
    records = []
    ids=sorted(set((ROOT/'splits/jaad_train_videos.txt').read_text().splitlines()+(ROOT/'splits/jaad_val_videos.txt').read_text().splitlines()))
    videos=[DATA/'JAAD_clips'/(name+'.mp4') for name in ids]
    assert len(videos)==311 and all(p.exists() for p in videos)
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8);workers=parser.parse_args().workers
    (ROOT/'runs/data_preparation').mkdir(parents=True,exist_ok=True)
    print(f'Extracting with {workers} video workers, one decoder thread per worker; original PNG encoding unchanged.', flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(extract_video, video) for video in videos]
        for future in as_completed(futures):
            records.append(future.result())
            records.sort(key=lambda r:r['video'])
            temporary=ROOT/'runs/data_preparation/.frame_progress.json.tmp'
            temporary.write_text(json.dumps(records,indent=2))
            temporary.replace(ROOT/'runs/data_preparation/frame_progress.json')
            frames=sum(r['frames'] for r in records)
            print(f'PROGRESS videos={len(records)}/311 frames={frames} elapsed={time.time()-started:.1f}s fps={frames/(time.time()-started):.2f}',flush=True)
    summary = {'videos': len(records), 'frames': sum(r['frames'] for r in records),
               'seconds': time.time() - started, 'png_bytes': sum(r['png_bytes'] for r in records),
               'workers': workers, 'decoder_threads_per_worker': 1}
    (ROOT / 'runs/data_preparation/frame_summary.json').write_text(json.dumps(summary, indent=2))
    print(summary, flush=True)

if __name__=='__main__':
    main()
