import os
from enum import Enum
from pydub import AudioSegment
import shutil
import traceback
import argparse
import math
import logging
from helpers import colors
from audio_separator.separator import Separator

def separate_audio(input_file_path: str, output_folder: str) -> None:
    """Separate vocals from audio with the audio_separator library."""

    output_names = {
        "Vocals": "vocals",
        "Instrumental": "no_vocals"
    }
    
    separator = Separator(
        output_dir=os.path.join(output_folder, "separated"),
        output_format="wav",
        log_level=logging.WARNING
    )
    
    separator.load_model(model_filename="mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt")
    separator.separate(input_file_path, output_names)

def copy_and_save_separated_audio(folder:str, song_name:str, txt_path:str, vocal_mix_volume:int=40) -> None:
    vocals_name = f"{song_name} [Vocals].mp3"
    instrumental_name = f"{song_name} [Instrumental].mp3"
    separated_audio_folder = os.path.join(folder, "separated")
    vocals_path = os.path.join(separated_audio_folder, "vocals.wav")
    instrumental_path = os.path.join(separated_audio_folder, "no_vocals.wav")
    
    # Convert vocals.wav to mp3
    if os.path.isfile(vocals_path):
      vocals_audio = AudioSegment.from_wav(vocals_path)
      vocals_audio.export(os.path.join(folder, vocals_name), format="mp3")

    # Convert no_vocals.wav to mp3 and add vocals at 20% volume
    if os.path.isfile(instrumental_path) and os.path.isfile(vocals_path):
      instrumental_audio = AudioSegment.from_wav(instrumental_path)
      
      # check if vocal_mix_volume is greater than 0, otherwise just export instrumental
      if vocal_mix_volume > 0:
        vocals_audio = AudioSegment.from_wav(vocals_path)
        
        # Reduce vocals volume
        percent_to_db = math.log10(vocal_mix_volume / 100) * 20
        vocals_reduced = vocals_audio + percent_to_db 
      
        # Mix the instrumental with the reduced vocals
        instrumental_with_backing = instrumental_audio.overlay(vocals_reduced)
        
      instrumental_with_backing.export(os.path.join(folder, instrumental_name), format="mp3")
    elif os.path.isfile(instrumental_path):
      # Fallback if vocals file doesn't exist
      instrumental_audio = AudioSegment.from_wav(instrumental_path)
      instrumental_audio.export(os.path.join(folder, instrumental_name), format="mp3")

    # Remove separated folder
    if os.path.isdir(separated_audio_folder):
      shutil.rmtree(separated_audio_folder)

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

def process_song_folder(input_folder:str, overwrite_existing:bool=False, vocals_volume:int=40) -> None:
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
      if not overwrite_existing:
        for line in lines:
          if line.strip().upper().startswith("#INSTRUMENTAL") or line.strip().upper().startswith("#VOCALS"):
            print(colors.light_blue_highlighted(f"⏩ Skipping folder '{os.path.basename(input_folder)}' as #INSTRUMENTAL or #VOCALS tag already exists."))
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
    
    audio_file_path = os.path.join(input_folder, audio_file)
    if not os.path.isfile(audio_file_path):
      raise FileNotFoundError(f"Audio file specified in the .txt file not found: {audio_file_path}")

    output_folder = input_folder

    separate_audio(audio_file_path, output_folder)
    copy_and_save_separated_audio(output_folder, os.path.splitext(os.path.basename(audio_file_path))[0], input_file, vocals_volume)


def main():
    
    parser=argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("--limit", help="Limits the amount of folders processed", type=int)
    parser.add_argument("--offset", help="Offset for the program", default=0, type=int)
    parser.add_argument("--overwrite", help="Overwrite existing files", action="store_true")
    parser.add_argument("--vocals_volume", help="Set vocals volume percentage (default is 40%)", type=int, default=40)
    
    
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
      print(colors.blue_highlighted(f"\n⏳({idx + 1} / {len(folder_list)}) Processing folder: {os.path.basename(folder)}"))
      try:
        process_song_folder(folder, overwrite_existing=args.overwrite, vocals_volume=args.vocals_volume)
        print(colors.bright_green_highlighted(f"✅ Finished processing folder: {os.path.basename(folder)}"))
      except Exception as e:
        print(colors.red_highlighted(f"🚩Error processing folder {folder}: {e}"))
        traceback.print_exc()
        
        
if __name__ == "__main__":
    main()