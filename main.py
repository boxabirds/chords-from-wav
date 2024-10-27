import librosa
import numpy as np
from madmom.audio.chroma import DeepChromaProcessor
from madmom.features.chords import CNNChordFeatureProcessor
from madmom.models import CRF
from music21 import stream, note, midi
import sys
import os

def validate_audio_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ['.wav', '.mp3']:
        raise ValueError(f"Unsupported file format. Please provide a .wav or .mp3 file. Got: {ext}")
    
    return file_path

def detect_chords(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file)

    # Extract chroma features
    proc = DeepChromaProcessor()
    chroma = proc(y, sr=sr)

    # Prepare chord recognition model
    feature_proc = CNNChordFeatureProcessor()
    crf = CRF(n_states=170, transitions='chord_transitions.txt', observation_model='CNNChordFeatures')
    crf.load('chord_crf_model.pkl')

    # Extract chord features
    features = feature_proc(chroma)
    
    # Recognize chords
    chords = crf(features)

    return chords

def chords_to_midi(chords, output_file):
    s = stream.Stream()

    for chord in chords:
        root_note = note.Note(chord.split(':')[0])
        bass_note = note.Note(chord.split(':')[1]) if ':' in chord else None
        
        chord_notes = [root_note]
        
        if bass_note:
            chord_notes.append(bass_note)
        
        # Add major third and perfect fifth
        chord_notes.append(note.Note(root_note.pitch.midi + 4))
        chord_notes.append(note.Note(root_note.pitch.midi + 7))

        chord_obj = chord.Chord(chord_notes)
        s.append(chord_obj)

    s.write('midi', fp=output_file)

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_audio_file>")
        print("Supported formats: .wav, .mp3")
        sys.exit(1)

    try:
        input_file = validate_audio_file(sys.argv[1])
        output_file = os.path.splitext(input_file)[0] + '_chords.mid'

        chords = detect_chords(input_file)
        chords_to_midi(chords, output_file)

        print(f"Detected chords and saved MIDI file: {output_file}")
    
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
