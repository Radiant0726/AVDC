import decord
from decord import VideoReader
from decord import cpu
import subprocess
import json
import soundfile as sf


def write_jsonl(path, item):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
# video_bins = [0, 30, 60, 120, 300, 600, 1800, 10000]
# question_bins = [0, 10, 50, 100, 200, 500, 2000]
# answer_bins = [0, 10, 50, 100, 200, 500, 2000]

video_bins = [0, 10, 20, 30, 40, 50, 61, 10000]
question_bins = [0, 25, 50, 100, 200, 500, 2000]
answer_bins = [0, 25, 50, 100, 200, 500, 2000]

def get_bin_labels(bins, name):
    return [f"{name}#{bins[i]}-{bins[i+1]}#" for i in range(len(bins)-1)] + [-1]

video_labels = get_bin_labels(video_bins, "video")
question_labels = get_bin_labels(question_bins, "q_len")
answer_labels = get_bin_labels(answer_bins, "a_len")

def assign_bin(value, bins, labels):
    if not value: 
        return None
    for i in range(len(bins) - 1):
        if bins[i] <= value < bins[i + 1]:
            return labels[i]
    if value >= bins[-1]:
        return labels[-1]
    return -1


def get_video_duration_ffprobe(video_path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        output = json.loads(result.stdout)
        duration = float(output["format"]["duration"])
        return duration
    except Exception as e:
        print(f"ffprobe error on {video_path}: {e}")
        return None

def get_video_duration_ffmpeg(file_path):
    """
    使用 ffmpeg 获取视频的时长（以秒为单位）。
    :param file_path: 视频文件的路径
    :return: 视频时长（秒），如果出错返回 None
    """
    # 构造 ffmpeg 命令
    command = [
        "ffmpeg", "-i", file_path,
        "-f", "null", "-"  # 输出到空设备，不生成实际文件
    ]

    try:
        # 执行命令并捕获输出
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        # 解析 stderr 输出以获取时长信息
        output = result.stderr
        # 查找 "Duration: HH:MM:SS.mmm" 这样的字符串
        duration_str = None
        for line in output.splitlines():
            if "Duration" in line:
                duration_str = line.split("Duration:")[1].strip().split(",")[0].strip()
                break

        if duration_str:
            # 将时长字符串转换为秒
            hours, minutes, seconds = map(float, duration_str.split(":"))
            total_seconds = hours * 3600 + minutes * 60 + seconds
            return total_seconds
        else:
            print("未找到时长信息")
            return None

    except subprocess.CalledProcessError as e:
        print(f"获取视频时长时出错: {e}")
        return None
    
def get_video_duration_decord(video_path):
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        duration = len(vr) / vr.get_avg_fps()
        return duration
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        return None


def get_audio_duration_sf(audio_path):
    with sf.SoundFile(audio_path) as f:
        duration = len(f) / f.samplerate
    return duration

import av
def check_if_video_has_audio(video_path):
    container = av.open(video_path)
    audio_streams = [stream for stream in container.streams if stream.type == "audio"]
    if not audio_streams:
        print("video has no audio")
        return False
    return True