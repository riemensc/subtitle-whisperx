#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Karaoke Maker mit Faster-Whisper
Effekt: Eine feste, kleine Untertitelzeile am unteren Rand.
Das aktive Wort färbt sich, der Rest bleibt weiß.
Fokus auf verschiedene kleine, dezente Schriftstile.
"""

import os
import argparse
import pysrt
from pysrt import SubRipFile, SubRipItem, SubRipTime 
import subprocess
import re
import json
import tempfile
import sys
import traceback
import logging
from datetime import datetime

WORDS_IN_CONTEXT_LINE = 7 

def setup_logging(log_file="karaoke_maker.log", debug_mode=False):
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logger = logging.getLogger()
    logger.setLevel(log_level)
    for handler in logger.handlers[:]: logger.removeHandler(handler)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level); file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level); console_handler.setFormatter(formatter)
    logger.addHandler(file_handler); logger.addHandler(console_handler)
    logging.info(f"Logging gestartet. Log-Datei: {log_file}, Debug-Modus: {debug_mode}")
    logging.info("=" * 80)

def check_faster_whisper():
    logging.info("=" * 50); logging.info("ÜBERPRÜFE FASTER-WHISPER"); logging.info("=" * 50)
    try:
        import faster_whisper
        logging.info(f"✓ 'faster_whisper' Modul importiert. Version: {faster_whisper.__version__}")
        logging.info(f"  Pfad: {faster_whisper.__file__}")
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            logging.info(f"✓ CUDA verfügbar: {cuda_available}")
            if cuda_available:
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "N/A"
                logging.info(f"  CUDA Geräte: {device_count}"); logging.info(f"  Primäres CUDA Gerät: {device_name}")
        except ImportError: logging.info("✗ PyTorch nicht installiert, CUDA-Status kann nicht überprüft werden")
        logging.info(f"Faster-Whisper ist korrekt installiert und einsatzbereit.")
        return True
    except ImportError as e:
        logging.error(f"✗ 'faster_whisper' konnte nicht importiert werden."); logging.error(f"  Fehlermeldung: {e}")
        logging.info("\nBitte installiere Faster-Whisper mit:\npip install faster-whisper"); return False

def extract_audio_from_video(video_path, output_audio_path=None):
    if output_audio_path is None:
        temp_dir = tempfile.gettempdir()
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        output_audio_path = os.path.join(temp_dir, f"{base_name}_{timestamp}.wav")
    logging.info(f"Extrahiere Audio aus: {video_path} nach {output_audio_path}")
    cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', output_audio_path, '-y']
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stderr: logging.debug(f"FFmpeg stderr: {process.stderr.decode('utf-8', errors='replace')}")
    if os.path.exists(output_audio_path):
        logging.info(f"Audio erfolgreich extrahiert nach {output_audio_path} ({os.path.getsize(output_audio_path)/(1024*1024):.2f} MB)")
    else: logging.error(f"Fehler bei Audio-Extraktion: Datei {output_audio_path} wurde nicht erstellt"); return None
    return output_audio_path

def extract_duration(video_path):
    cmd = ['ffmpeg', '-i', video_path, '-f', 'null', '-']
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    duration_match = re.search(r'Duration: (\d{2}):(\d{2}):(\d{2})\.(\d{2})', result.stderr)
    if duration_match:
        h, m, s, cs = map(int, duration_match.groups())
        return (h * 3600 + m * 60 + s) * 1000 + cs * 10
    logging.warning("Video-Dauer konnte nicht ermittelt werden, verwende 60 Sekunden"); return 60000

def srt_time_to_string(srt_time_obj):
    if not isinstance(srt_time_obj, SubRipTime):
        logging.error(f"srt_time_to_string: Ungültiges Objekt erhalten: {type(srt_time_obj)}")
        return "00:00:00,000" 
    h = int(srt_time_obj.hours)
    m = int(srt_time_obj.minutes)
    s = int(srt_time_obj.seconds)
    ms = int(srt_time_obj.milliseconds)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def transcribe_with_faster_whisper(video_path, model_size="medium", language=None, min_segment_length=0.1):
    logging.info(f"Starte Transkription mit Faster-Whisper (Modell: {model_size}, Sprache: {language or 'auto'})")
    try: from faster_whisper import WhisperModel; import torch
    except ImportError as e: logging.error(f"Fehler beim Import von Faster-Whisper: {e}"); logging.error("Bitte installiere mit: pip install faster-whisper"); raise
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    logging.info(f"Verwende {device.upper()} mit Compute-Typ {compute_type} für Transkription.")

    audio_path = extract_audio_from_video(video_path)
    if not audio_path: raise RuntimeError("Audio-Extraktion fehlgeschlagen.")
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logging.info(f"Whisper-Modell '{model_size}' geladen. Starte Transkription von {audio_path}...")
        
        segments_generator, info = model.transcribe(audio_path, beam_size=5, vad_filter=True, word_timestamps=True, language=language)
        logging.info(f"Erkannte Sprache: {info.language} (Wahrscheinlichkeit: {info.language_probability:.2f})")
        
        segments_list = []
        if segments_generator:
            segments_list = list(segments_generator)

        if not segments_list:
            logging.warning("!!! transcribe_with_faster_whisper: 'segments'-Generator von model.transcribe ergab eine leere Liste!!!")
        elif logging.getLogger().isEnabledFor(logging.DEBUG): 
             for i, seg in enumerate(segments_list):
                if i < 5: 
                    logging.debug(f"Debug transcribe_with_faster_whisper: Segment {i} - Text: '{seg.text}', Start: {seg.start}, Ende: {seg.end}")
                    if hasattr(seg, 'words') and seg.words:
                        logging.debug(f"  Segment {i} Wörter (erste 3): {[(w.word, w.start, w.end) for w in seg.words[:3]]}")
                    else:
                        logging.debug(f"  Segment {i} hat keine Wörter.")
        
        return create_word_level_srt(segments_list, min_segment_length) 
    finally:
        if audio_path and os.path.exists(audio_path):
            try: os.remove(audio_path); logging.debug(f"Temporäre Audio-Datei gelöscht: {audio_path}")
            except Exception as e: logging.warning(f"Konnte temporäre Audio-Datei nicht löschen: {audio_path} - {e}")

def create_word_level_srt(segments_list, min_segment_length=0.1):
    logging.info(f"Erstelle SRT mit einem Wort pro Segment, min_segment_length: {min_segment_length}s")
    srt_blocks = [] 
    item_index = 1
    
    if not segments_list:
        logging.warning("create_word_level_srt: Empfangene Segmentliste ist leer.")
    
    for segment_idx, segment in enumerate(segments_list):
        if not hasattr(segment, 'words') or not segment.words:
            logging.debug(f"Segment {segment_idx} hat keine Wörter oder kein 'words'-Attribut.")
            continue
        for word_info in segment.words:
            start, end = word_info.start, word_info.end
            if start is None or end is None: 
                logging.warning(f"Wort '{word_info.word}' hat ungültige Zeitstempel (start={start}, end={end}), wird übersprungen.")
                continue
            if not (isinstance(start, (int, float)) and isinstance(end, (int, float))):
                logging.warning(f"Wort '{word_info.word}' hat nicht-numerische Zeitstempel (start={type(start)}, end={type(end)}), wird übersprungen.")
                continue
            if end < start: 
                logging.warning(f"Korrigiere Endzeit für Wort '{word_info.word}': Ende ({end}) < Start ({start}). Setze minimale Dauer.")
                end = start + 0.01
            
            duration = end - start
            if duration < min_segment_length:
                end = start + min_segment_length
            
            text = word_info.word.strip()
            if not text: continue

            start_time = SubRipTime(seconds=float(start)) 
            end_time = SubRipTime(seconds=float(end))   
            
            block = f"{item_index}\n"
            block += f"{srt_time_to_string(start_time)} --> {srt_time_to_string(end_time)}\n"
            block += f"{text}\n\n" 
            srt_blocks.append(block)
            item_index += 1
            
    if not srt_blocks:
        logging.warning("Keine SRT-Blöcke erstellt aus den Segmenten.")
        return "" 

    final_srt_string = "".join(srt_blocks) 
    logging.debug(f"Erstellter SRT-String (manuell, erste 300 Zeichen): '{final_srt_string[:300]}...'")
    return final_srt_string

def create_dummy_srt_content(video_duration, text=None):
    if text is None: text = "ICH SAGE IHR NICHT DASS ES KEIN"
    words = text.split()
    if not words: logging.warning("Dummy-Text ist leer."); return "1\n00:00:00,000 --> 00:00:01,000\n \n\n"
    duration_per_word_ms = (video_duration / len(words)) if video_duration > 0 else 1000.0
    
    srt_blocks = []
    current_time_ms = 0
    for i, word in enumerate(words):
        start_ms = current_time_ms
        end_ms = current_time_ms + duration_per_word_ms
        
        start_time = SubRipTime(milliseconds=int(start_ms))
        end_time = SubRipTime(milliseconds=int(end_ms))

        block = f"{i + 1}\n"
        block += f"{srt_time_to_string(start_time)} --> {srt_time_to_string(end_time)}\n"
        block += f"{word}\n\n" 
        srt_blocks.append(block)
        current_time_ms = end_ms
        
    final_srt_string = "".join(srt_blocks)
    logging.debug(f"Erstellter Dummy-SRT-String (manuell, erste 200 Zeichen): {final_srt_string[:200]}...")
    return final_srt_string

def format_time_ass(subriptime_obj):
    if subriptime_obj is None: return "0:00:00.00"
    return f"{subriptime_obj.hours:01d}:{subriptime_obj.minutes:02d}:{subriptime_obj.seconds:02d}.{subriptime_obj.milliseconds//10:02d}"

def create_stylized_ass_file(srt_content, ass_path, style_type="tiktok"):
    global WORDS_IN_CONTEXT_LINE
    logging.info(f"Erstelle stilisierte ASS-Datei mit Stil '{style_type}' (Feste Zeile mit \\kf-Karaoke, {WORDS_IN_CONTEXT_LINE} Wörter/Zeile)")
    
    if not srt_content or not srt_content.strip():
        logging.error("create_stylized_ass_file: srt_content ist leer. Erstelle leere ASS.")
        with open(ass_path, "w", encoding="utf-8") as f:
             f.write("[Script Info]\nScriptType: v4.00+\n[V4+ Styles]\nStyle: Default,Arial,20,&Hffffff,&Hffffff,&H0,&H0,0,0,0,0,100,100,0,0,1,1,0,2,0,0,0,0\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        return ass_path

    try:
        subs = SubRipFile.from_string(srt_content)
        logging.info(f"SRT-Inhalt erfolgreich geparst. Anzahl Wörter für ASS-Generierung: {len(subs)}")
    except Exception as e:
        logging.error(f"Fehler beim Parsen des SRT-Inhalts für ASS: {e}"); logging.error(traceback.format_exc())
        with open(ass_path, "w", encoding="utf-8") as f:
             f.write("[Script Info]\nScriptType: v4.00+\n[V4+ Styles]\nStyle: Default,Arial,20,&Hffffff,&Hffffff,&H0,&H0,0,0,0,0,100,100,0,0,1,1,0,2,0,0,0,0\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        return ass_path

    if not subs:
        logging.error("Keine Untertitel-Items (subs) zum Verarbeiten für ASS gefunden.")
        with open(ass_path, "w", encoding="utf-8") as f:
             f.write("[Script Info]\nScriptType: v4.00+\n[V4+ Styles]\nStyle: Default,Arial,20,&Hffffff,&Hffffff,&H0,&H0,0,0,0,0,100,100,0,0,1,1,0,2,0,0,0,0\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        return ass_path

    styles = {
        "social_fixed_yellow": { 
            "font": "Arial", "size": 18, "bold": 1, 
            "unsung_color": "&H00FFFFFF", "sung_color": "&H0000FFFF",        
            "outline_color": "&HAA000000", "outline_width": 1, 
            "shadow": 0, "position": 2, "margin_v": 15                     
        },
        "social_fixed_green": {
            "font": "Arial", "size": 18, "bold": 1, 
            "unsung_color": "&H00FFFFFF", "sung_color": "&H0000FF00",  
            "outline_color": "&HAA000000", "outline_width": 1, "shadow": 0, "position": 2, "margin_v": 15
        },
        "subtle_highlight_orange": { 
            "font": "Arial", "size": 22, "bold": 0, # Leicht größer als 18
            "unsung_color": "&H00F0F0F0", "sung_color": "&H0000B4FF",    
            "outline_color": "&H9A000000", "outline_width": 1, "shadow": 1, "position": 2, "margin_v": 20
        },
        "clean_blue": {
            "font": "Calibri", "size": 20, "bold": 0, 
            "unsung_color": "&H00FFFFFF", "sung_color": "&H00FF7F00",    
            "outline_color": "&HA0000000", "outline_width": 1, "shadow": 0, "position": 2, "margin_v": 18
        },
        "minimal_cyan": {
            "font": "Arial", "size": 16, "bold": 0, 
            "unsung_color": "&H00E0E0E0", "sung_color": "&H00FFFF00",    
            "outline_color": "&H70000000", "outline_width": 1, "shadow": 0, "position": 2, "margin_v": 12 
        },
        "warm_red_highlight": {
            "font": "Verdana", "size": 20, "bold": 0, 
            "unsung_color": "&H00FFFFFF", "sung_color": "&H000000FF",    
            "outline_color": "&HA0000000", "outline_width": 1, "shadow": 1, "position": 2, "margin_v": 18
        },
        "simple_bold_yellow_small": { 
            "font": "Arial", "size": 20, "bold": 1,
            "unsung_color": "&H00FFFFFF", "sung_color": "&H0000FFFF", 
            "outline_color": "&HAA000000", "outline_width": 1, "shadow": 0, "position": 2, "margin_v": 18
        }
    }
    
    if style_type not in styles: 
        logging.warning(f"Ungültiger Stil '{style_type}'. Verwende 'social_fixed_yellow'.")
        style_type = "social_fixed_yellow" 
    
    style_params = styles[style_type]
    logging.debug(f"Verwende Stil-Konfiguration für \\kf-Karaoke: {style_params}")
    
    ass_header = f"""[Script Info]
Title: Karaoke Subtitles by Script; ScriptType: v4.00+; PlayResX: 1280; PlayResY: 720; ScaledBorderAndShadow: yes; WrapStyle: 0 
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: KaraokeStyle,{style_params["font"]},{style_params["size"]},{style_params["unsung_color"]},{style_params["sung_color"]},{style_params["outline_color"]},{style_params["outline_color"]},{style_params["bold"]},0,0,0,100,100,0,0,1,{style_params["outline_width"]},{style_params["shadow"]},{style_params["position"]},30,30,{style_params.get("margin_v", 25)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    all_words_list = list(subs) 
    dialogue_entries = []
    current_line_buffer = []
    line_start_time = None

    for i, word_item in enumerate(all_words_list):
        if not isinstance(word_item.start, SubRipTime) or not isinstance(word_item.end, SubRipTime):
            logging.warning(f"Ungültige Zeit-Objekte für Wort '{word_item.text}', überspringe."); continue

        if line_start_time is None: line_start_time = word_item.start
        current_line_buffer.append(word_item)

        if len(current_line_buffer) >= WORDS_IN_CONTEXT_LINE or i == len(all_words_list) - 1:
            if not current_line_buffer: continue
            line_end_time = current_line_buffer[-1].end
            karaoke_text_parts = []
            for word_in_line in current_line_buffer:
                word_text_cleaned = word_in_line.text.strip()
                if not word_text_cleaned: continue
                duration_srt_time = word_in_line.end - word_in_line.start
                duration_seconds = (duration_srt_time.hours * 3600 + duration_srt_time.minutes * 60 + duration_srt_time.seconds + duration_srt_time.milliseconds / 1000.0)
                duration_cs = max(1, int(round(duration_seconds * 100)))
                karaoke_text_parts.append(f"{{\\kf{duration_cs}}}{word_text_cleaned}")
            full_karaoke_text = " ".join(karaoke_text_parts)
            dialogue_start_str = format_time_ass(line_start_time)
            dialogue_end_str = format_time_ass(line_end_time)
            dialogue_entries.append(f"Dialogue: 0,{dialogue_start_str},{dialogue_end_str},KaraokeStyle,,0,0,0,,{full_karaoke_text}\n")
            current_line_buffer = []
            line_start_time = None

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_header)
        for entry in dialogue_entries:
            f.write(entry)
    
    logging.info(f"ASS-Datei erfolgreich erstellt: {ass_path}")
    logging.info(f"Erstellt mit {len(dialogue_entries)} Karaoke-Zeilen (\\kf-Effekt).")
    return ass_path

def apply_subtitles_to_video(video_path, srt_content, output_video_path, style_type="tiktok"):
    # ... (bleibt gleich)
    ass_fd, temp_ass_path = tempfile.mkstemp(suffix=".ass")
    os.close(ass_fd) 
    try:
        create_stylized_ass_file(srt_content, temp_ass_path, style_type)
        if sys.platform == "win32":
            ffmpeg_filter_path = temp_ass_path.replace('\\', '/')
            ffmpeg_filter_path = ffmpeg_filter_path.replace(':', '\\:')
        else:
            ffmpeg_filter_path = temp_ass_path.replace(':', '\\:')
        ffmpeg_ass_path_arg = f"ass='{ffmpeg_filter_path}'"
        command_parts = ['ffmpeg', '-i', video_path, '-vf', ffmpeg_ass_path_arg, '-c:v', 'libx264', '-c:a', 'aac', '-preset', 'fast', '-crf', '23', output_video_path, '-y']
        logging.info(f"Füge Untertitel zum Video hinzu...")
        logging.debug(f"FFmpeg-Befehl: {' '.join(command_parts)}")
        process = subprocess.run(command_parts, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE) 
        stderr_output = process.stderr.decode('utf-8', errors='replace') if process.stderr else ""
        if stderr_output:
            if process.returncode != 0 or logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(f"FFmpeg stderr Output:\n{stderr_output}")
            elif process.returncode == 0 and not logging.getLogger().isEnabledFor(logging.DEBUG):
                for line in stderr_output.splitlines():
                    if "error" in line.lower() or "warning" in line.lower(): logging.info(f"FFmpeg (Info): {line}"); break 
        if process.returncode != 0:
            logging.error(f"FFmpeg-Befehl fehlgeschlagen mit Rückgabecode {process.returncode}")
            if not logging.getLogger().isEnabledFor(logging.DEBUG): logging.error(f"Fehlermeldung (Auszug): {stderr_output[:1000]}...")
            raise RuntimeError(f"FFmpeg-Befehl fehlgeschlagen. Siehe Log für Details.")
        if os.path.exists(output_video_path):
            video_size = os.path.getsize(output_video_path) / (1024 * 1024)
            logging.info(f"Video erfolgreich erstellt: {output_video_path} ({video_size:.2f} MB)")
        else:
            logging.error(f"FFmpeg-Befehl wurde ohne Fehler ausgeführt, aber die Ausgabedatei {output_video_path} wurde nicht erstellt")
            raise FileNotFoundError(f"Ausgabevideo nicht gefunden: {output_video_path}")
    except Exception as e: logging.error(f"Fehler beim Hinzufügen der Untertitel: {e}"); logging.error(traceback.format_exc()); raise
    finally:
        if os.path.exists(temp_ass_path):
            try: os.remove(temp_ass_path); logging.debug(f"Temporäre ASS-Datei gelöscht: {temp_ass_path}")
            except Exception as e_rem_ass: logging.warning(f"Konnte temporäre ASS-Datei nicht löschen {temp_ass_path}: {e_rem_ass}")
    return output_video_path


def generate_karaoke_video(video_path, output_video_path=None, style_type="tiktok", 
                          custom_text=None, model_size="medium", language=None, min_segment_length=0.1):
    # ... (bleibt gleich)
    if output_video_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        safe_base_name = "".join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in base_name).rstrip()
        output_video_path = f"karaoke_{safe_base_name}_{style_type}_{timestamp}.mp4"
    logging.info(f"Starte Verarbeitung von: {video_path}"); logging.info(f"Ausgabedatei: {output_video_path}"); logging.info(f"Verwende Stil: {style_type}")
    srt_content = "" 
    try:
        if custom_text:
            logging.info(f"Verwende benutzerdefinierten Text: \"{custom_text[:50]}...\"")
            video_duration = extract_duration(video_path)
            srt_content = create_dummy_srt_content(video_duration, custom_text)
        else:
            logging.info("Starte Transkription mit Faster-Whisper...")
            try:
                srt_content = transcribe_with_faster_whisper(video_path, model_size=model_size, language=language, min_segment_length=min_segment_length)
                if not srt_content: 
                    logging.warning("SRT-Inhalt ist leer nach Transkription.")
                    raise ValueError("Transkription ergab leeren SRT-Inhalt.")
            except Exception as e:
                logging.error(f"FEHLER bei der Transkription: {e}"); logging.error(traceback.format_exc())
                logging.info("Fallback zu Dummy-Text für die Untertitel...")
                video_duration = extract_duration(video_path); srt_content = create_dummy_srt_content(video_duration, "TRANSKRIPTION FEHLGESCHLAGEN")
        final_srt_path = output_video_path.replace('.mp4', '.final.srt')
        try:
            with open(final_srt_path, 'w', encoding='utf-8') as f: f.write(srt_content) 
            logging.info(f"Finaler SRT-Inhalt (vor ASS) gespeichert in: {final_srt_path}")
        except Exception as e_write: logging.warning(f"Konnte finalen SRT-Inhalt nicht speichern: {e_write}")
        if not srt_content.strip(): 
            logging.error("SRT-Inhalt ist komplett leer. Videoerstellung wird abgebrochen."); 
            raise ValueError("SRT-Inhalt ist leer.")
        logging.info("Füge Untertitel zum Video hinzu...")
        final_video_path = apply_subtitles_to_video(video_path, srt_content, output_video_path, style_type)
        if final_video_path and os.path.exists(final_video_path):
            logging.info(f"Video erfolgreich erstellt. Dateigröße: {os.path.getsize(final_video_path)/(1024*1024):.2f} MB")
        else: logging.warning(f"WARNUNG: Ausgabevideo wurde nicht gefunden: {final_video_path}")
        logging.info(f"Fertig! Karaoke-Video wurde erstellt: {final_video_path}")
        return final_video_path
    except Exception as e:
        logging.error("=" * 50); logging.error(f"KRITISCHER FEHLER: {e}"); logging.error(traceback.format_exc()); logging.error("=" * 50)
        return None

def main():
    global WORDS_IN_CONTEXT_LINE 
    available_styles = ["social_fixed_yellow", "social_fixed_green", "subtle_highlight_orange", "clean_blue", "minimal_cyan", "warm_red_highlight", "simple_bold_yellow_small", "tiktok_fixed"] 
    parser = argparse.ArgumentParser(description="Karaoke Maker", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--video", "-v", required=True, help="Eingabe-Videodatei")
    parser.add_argument("--output", "-o", help="Ausgabe-Videodatei")
    parser.add_argument("--style", "-s", choices=available_styles, default="social_fixed_yellow", help=f"Stil (Standard: social_fixed_yellow).\nVerfügbar: {', '.join(available_styles)}")
    parser.add_argument("--text", "-t", help="Benutzerdefinierter Text")
    parser.add_argument("--model", "-m", default="medium", help="Whisper-Modell")
    parser.add_argument("--language", "-l", help="Sprachcode (z.B. 'de', 'en')")
    parser.add_argument("--min-segment-length", type=float, default=0.1, help="Min. Dauer Wort-Segment (SRT) in Sek.")
    parser.add_argument("--words-per-line", type=int, default=WORDS_IN_CONTEXT_LINE, help=f"Wörter pro Karaoke-Zeile (Std: {WORDS_IN_CONTEXT_LINE})")
    parser.add_argument("--debug", "-d", action="store_true", help="Debug-Ausgaben")
    parser.add_argument("--log", help="Log-Datei")
    parser.add_argument("--check", "-c", action="store_true", help="Installation prüfen")
    args = parser.parse_args()
    WORDS_IN_CONTEXT_LINE = args.words_per_line
    log_file_path = args.log if args.log else os.path.join(os.getcwd(), "karaoke_maker.log")
    setup_logging(log_file=log_file_path, debug_mode=args.debug)
    logging.info("=" * 80 + f"\nKARAOKE MAKER GESTARTET\n" + "=" * 80)
    logging.info(f"Argumente: {vars(args)}")

    if args.check:
        logging.info("Starte Installationsüberprüfung..."); ffmpeg_ok = False
        try: result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True); logging.info(f"✓ FFmpeg Version: {result.stdout.split(chr(10))[0]}"); ffmpeg_ok = True
        except FileNotFoundError: logging.error("✗ FFmpeg nicht gefunden.")
        except Exception as e: logging.error(f"✗ FFmpeg Fehler: {e}")
        whisper_ok = check_faster_whisper()
        if ffmpeg_ok and whisper_ok: logging.info("✓ Alle Hauptkomponenten scheinen korrekt installiert zu sein.")
        else: logging.error("✗ Einige Komponenten sind nicht korrekt installiert."); 
        sys.exit(0)
        
    if not os.path.exists(args.video): logging.error(f"Videodatei nicht gefunden: {args.video}"); sys.exit(1)
    
    karaoke_video = generate_karaoke_video(video_path=args.video, output_video_path=args.output, style_type=args.style, custom_text=args.text, model_size=args.model, language=args.language, min_segment_length=args.min_segment_length)
    
    if karaoke_video and os.path.exists(karaoke_video):
        logging.info(f"\nKaraoke-Video erfolgreich erstellt: {os.path.abspath(karaoke_video)}")
        print(f"\nKaraoke-Video erfolgreich erstellt: {os.path.abspath(karaoke_video)}")
    else:
        logging.error("Karaoke-Video wurde nicht erstellt."); sys.exit(1)

if __name__ == "__main__":
    main()