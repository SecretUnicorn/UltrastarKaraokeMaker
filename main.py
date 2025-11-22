import os
from enum import Enum
from pydub import AudioSegment
import shutil
import sys
import argparse
from helpers import colors
import demucs.separate

class DemucsModel(Enum):
    HTDEMUCS = "htdemucs"           # first version of Hybrid Transformer Demucs. Trained on MusDB + 800 songs. Default model.
    HTDEMUCS_FT = "htdemucs_ft"     # fine-tuned version of htdemucs, separation will take 4 times more time but might be a bit better. Same training set as htdemucs.
    HTDEMUCS_6S = "htdemucs_6s"     # 6 sources version of htdemucs, with piano and guitar being added as sources. Note that the piano source is not working great at the moment.
    HDEMUCS_MMI = "hdemucs_mmi"     # Hybrid Demucs v3, retrained on MusDB + 800 songs.
    MDX = "mdx"                     # trained only on MusDB HQ, winning model on track A at the MDX challenge.
    MDX_EXTRA = "mdx_extra"         # trained with extra training data (including MusDB test set), ranked 2nd on the track B of the MDX challenge.
    MDX_Q = "mdx_q"                 # quantized version of the previous models. Smaller download and storage but quality can be slightly worse.
    MDX_EXTRA_Q = "mdx_extra_q"     # quantized version of mdx_extra. Smaller download and storage but quality can be slightly worse.
    SIG = "SIG"                     # Placeholder for a single model from the model zoo.


def check_file_exists(file_path: str) -> bool:
    """Checks if a file exists."""
    return os.path.isfile(file_path)

def separate_audio(input_file_path: str, output_folder: str, model: DemucsModel, device="cpu") -> None:
    """Separate vocals from audio with demucs."""

    print(
        f"""
        =====================
        STARTING AUDIO SEPARATION
        =====================
        Input file: {input_file_path}
        Output folder: {output_folder}
        Model: {model.value}
        Device: {device}
        """
    )

    demucs.separate.main(
        [
            "--two-stems", "vocals",
            "-d", f"{device}",
            "--float32",
            "-n",
            model.value,
            "--out", f"{os.path.join(output_folder, 'separated')}",
            f"{input_file_path}",
        ]
    )

def separate_vocal_from_audio(cache_folder_path: str,
                              audio_output_file_path: str,
                              use_separated_vocal: bool,
                              create_karaoke: bool,
                              pytorch_device: str,
                              model: DemucsModel,
                              skip_cache: bool = False) -> str:
    """Separate vocal from audio"""
    demucs_output_folder = os.path.splitext(os.path.basename(audio_output_file_path))[0]
    audio_separation_path = os.path.join(cache_folder_path, "separated", model.value, demucs_output_folder)

    vocals_path = os.path.join(audio_separation_path, "vocals.wav")
    instrumental_path = os.path.join(audio_separation_path, "no_vocals.wav")
    if use_separated_vocal or create_karaoke:
        cache_available = check_file_exists(vocals_path) and check_file_exists(instrumental_path)
        if skip_cache or not cache_available:
            separate_audio(audio_output_file_path, cache_folder_path, model, pytorch_device)
        else:
            print(f"cache > reusing cached separated vocals")

    return audio_separation_path

def copy_and_save_serpated_audio(folder, song_name, txt_path) -> None:
    vocals_name = f"{song_name} [Vocals].mp3"
    instrumental_name = f"{song_name} [Instrumental].mp3"
    seperation_folder = os.path.join(folder, "separated", "htdemucs", song_name)
    vocals_path = os.path.join(seperation_folder, "vocals.wav")
    instrumental_path = os.path.join(seperation_folder, "no_vocals.wav")
    # Convert vocals.wav to mp3
    if os.path.isfile(vocals_path):
      vocals_audio = AudioSegment.from_wav(vocals_path)
      vocals_audio.export(os.path.join(folder, vocals_name), format="mp3")

    # Convert no_vocals.wav to mp3
    if os.path.isfile(instrumental_path):
      instrumental_audio = AudioSegment.from_wav(instrumental_path)
      instrumental_audio.export(os.path.join(folder, instrumental_name), format="mp3")

    # Remove separated folder
    seperation_base_folder = os.path.join(folder, "separated")
    if os.path.isdir(seperation_base_folder):
      shutil.rmtree(seperation_base_folder)

    # Add vocals and instrumental to txt file
    with open(txt_path, "r", encoding="utf-8") as f:
      lines = f.readlines()

    new_lines = []
    vocals_line = f"#VOCALS:{vocals_name}\n"
    instrumental_line = f"#INSTRUMENTAL:{instrumental_name}\n"
    inserted = False

    for line in lines:
      new_lines.append(line)
      if not inserted and line.strip().upper().startswith("#MP3"):
        new_lines.append(vocals_line)
        new_lines.append(instrumental_line)
        inserted = True

    with open(txt_path, "w", encoding="utf-8") as f:
      f.writelines(new_lines)

def process_song_folder(input_folder:str) -> None:
    # search for .txt file in input_folder
    input_file = None
    for file in os.listdir(input_folder):
      if file.lower().endswith(".txt"):
        input_file = os.path.join(input_folder, file)
        break

    if input_file is None:
      raise FileNotFoundError("No .txt file found in the input folder.")

    # read file and search for #MP3 or #AUDIO tag
    audio_file = None
    # Check if #INSTRUMENTAL or #VOCALS is already in the file, skip if found
    with open(input_file, "r", encoding="utf-8") as f:
      lines = f.readlines()
      for line in lines:
        if line.strip().upper().startswith("#INSTRUMENTAL") or line.strip().upper().startswith("#VOCALS"):
          print(colors.light_blue_highlighted(f"Skipping folder '{input_folder}' as #INSTRUMENTAL or #VOCALS tag already exists."))
          return
      for line in lines:
        if line.startswith("#MP3"):
          audio_file = line[5:].strip()
          break
        elif line.startswith("#AUDIO"):
          audio_file = line[7:].strip()
          break
    if audio_file is None:
      raise ValueError("No #MP3 or #AUDIO tag found in the .txt file.")
    
    print("Joining path:", input_folder, "with", audio_file)
    audio_file_path = os.path.join(input_folder, audio_file)
    if not os.path.isfile(audio_file_path):
      raise FileNotFoundError(f"Audio file specified in the .txt file not found: {audio_file_path}")

    output_folder = input_folder
    model = DemucsModel.HTDEMUCS
    device = "cpu"  # Using CPU since RTX 5080 requires newer PyTorch

    separate_audio(audio_file_path, output_folder, model, device)
    copy_and_save_serpated_audio(output_folder, os.path.splitext(os.path.basename(audio_file_path))[0], input_file)

def main():
    
    parser=argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("--limit", help="Limits the amount of folders processed")
    parser.add_argument("--offset", help="Offset for the program")
    
    # take base folder from argument
    
    
    args=parser.parse_args()
    # get all folders inside of input_folder
    folder_list = [os.path.join(args.input_folder, name) for name in os.listdir(args.input_folder) if os.path.isdir(os.path.join(args.input_folder, name))]
    offset = int(args.offset) if args.offset else 0
    if args.limit and offset > 0:
      folder_list = folder_list[offset:offset + int(args.limit)]
    elif args.limit and not offset:
      folder_list = folder_list[:int(args.limit)]
    elif offset > 0:
      folder_list = folder_list[offset:]
    
    for idx, folder in enumerate(folder_list):
      print(colors.blue_highlighted(f"({idx + 1} / {len(folder_list)}) Processing folder: {folder}"))
      try:
        process_song_folder(folder)
        print(colors.bright_green_highlighted(f"Finished processing folder: {folder}"))
      except Exception as e:
        print(colors.red_highlighted(f"Error processing folder {folder}: {e}"))
        
if __name__ == "__main__":
    main()