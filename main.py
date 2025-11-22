import os
import traceback
import argparse
from helpers import colors
from helpers.audio_seperator import AudioSeparator, DemucsModel


def process_song_folder(input_folder:str, device:str="cuda", overwrite_existing:bool=False, vocals_volume:int=40) -> None:
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

    # Initialize separator with CPU device
    separator = AudioSeparator(device=device)
    
    # Separate audio and create karaoke tracks using HTDEMUCS model
    separator.separate_audio(audio_file_path, input_folder, DemucsModel.HTDEMUCS)
    separator.copy_and_save_separated_audio(input_folder, os.path.splitext(os.path.basename(audio_file_path))[0], input_file, vocals_volume)


def main():
    
    parser=argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("--device", help="Device to use for separation (cpu or cuda)", type=str, required=True)
    parser.add_argument("--limit", help="Limits the amount of folders processed", type=int)
    parser.add_argument("--offset", help="Offset for the program", default=0, type=int)
    parser.add_argument("--overwrite", help="Overwrite existing files", action="store_true")
    parser.add_argument("--vocals_volume", help="Set vocals volume percentage (default is 40%%)", type=int, default=40)
    
    
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
        process_song_folder(folder, args.device, overwrite_existing=args.overwrite, vocals_volume=args.vocals_volume)
        print(colors.bright_green_highlighted(f"✅ Finished processing folder: {os.path.basename(folder)}"))
      except Exception as e:
        print(colors.red_highlighted(f"🚩Error processing folder {folder}: {e}"))
        traceback.print_exc()
        
        
if __name__ == "__main__":
    main()