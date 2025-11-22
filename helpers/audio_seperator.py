import os
import logging
import shutil
import math
from enum import Enum
from pydub import AudioSegment
from typing import Literal

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

class AudioSeparator:
    def __init__(self, device: Literal["cuda", "cpu"] = "cpu"):
        """
        Initialize the AudioSeparator with the specified device.
        
        Args:
            device: Either "cuda" for GPU acceleration or "cpu" for CPU processing
        """
        self.device = device
        
    def separate_audio(self, input_file_path: str, output_folder: str, model: DemucsModel = DemucsModel.HTDEMUCS) -> None:
        """
        Separate vocals from audio using the appropriate method based on device.
        
        Args:
            input_file_path: Path to the input audio file
            output_folder: Folder where separated audio will be saved
            model: DemucsModel to use for separation (only used for CPU mode)
        """
        if self.device == "cuda":
            self._separate_with_audio_separator(input_file_path, output_folder)
        else:
            self._separate_with_demucs(input_file_path, output_folder, model)
    
    def _separate_with_audio_separator(self, input_file_path: str, output_folder: str) -> None:
        """Separate using audio_separator library (GPU optimized)."""
        from audio_separator.separator import Separator
        
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
    
    def _separate_with_demucs(self, input_file_path: str, output_folder: str, model: DemucsModel) -> None:
        """Separate using demucs library (CPU compatible)."""
        import demucs.separate
        
        demucs.separate.main([
            "--two-stems", "vocals",
            "-d", self.device,
            "--float32",
            "-n", model.value,
            "--out", os.path.join(output_folder, "separated"),
            input_file_path,
        ])
    
    def copy_and_save_separated_audio(self, folder: str, song_name: str, txt_path: str, vocal_mix_volume: int = 40) -> None:
        """
        Convert separated audio files to MP3 and create karaoke track with backing vocals.
        
        Args:
            folder: Folder containing the separated audio
            song_name: Name of the song (used for output filenames)
            txt_path: Path to the .txt file to update
            vocal_mix_volume: Volume percentage for backing vocals in instrumental track
        """
        vocals_name = f"{song_name} [Vocals].mp3"
        instrumental_name = f"{song_name} [Instrumental].mp3"
        
        if self.device == "cuda":
            # audio_separator creates files directly in separated folder
            separated_audio_folder = os.path.join(folder, "separated")
            vocals_path = os.path.join(separated_audio_folder, "vocals.wav")
            instrumental_path = os.path.join(separated_audio_folder, "no_vocals.wav")
        else:
            # demucs creates nested folder structure
            separated_audio_folder = os.path.join(folder, "separated", "htdemucs", song_name)
            vocals_path = os.path.join(separated_audio_folder, "vocals.wav")
            instrumental_path = os.path.join(separated_audio_folder, "no_vocals.wav")
        
        # Convert vocals.wav to mp3
        if os.path.isfile(vocals_path):
            vocals_audio = AudioSegment.from_wav(vocals_path)
            vocals_audio.export(os.path.join(folder, vocals_name), format="mp3")

        # Convert instrumental and optionally add backing vocals
        if os.path.isfile(instrumental_path):
            instrumental_audio = AudioSegment.from_wav(instrumental_path)
            
            # Add backing vocals if volume > 0 and vocals exist
            if vocal_mix_volume > 0 and os.path.isfile(vocals_path):
                vocals_audio = AudioSegment.from_wav(vocals_path)
                
                # Calculate dB reduction for the specified percentage
                percent_to_db = math.log10(vocal_mix_volume / 100) * 20
                vocals_reduced = vocals_audio + percent_to_db 
                
                # Mix the instrumental with the reduced vocals
                instrumental_with_backing = instrumental_audio.overlay(vocals_reduced)
                instrumental_with_backing.export(os.path.join(folder, instrumental_name), format="mp3")
            else:
                # Pure instrumental without backing vocals
                instrumental_audio.export(os.path.join(folder, instrumental_name), format="mp3")

        # Clean up separated folder
        separated_base_folder = os.path.join(folder, "separated")
        if os.path.isdir(separated_base_folder):
            shutil.rmtree(separated_base_folder)

        # Update .txt file with new tags
        self._update_txt_file(txt_path, vocals_name, instrumental_name)
    
    def _update_txt_file(self, txt_path: str, vocals_name: str, instrumental_name: str) -> None:
        """Add vocals and instrumental tags to the .txt file."""
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