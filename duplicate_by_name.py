import os
import shutil
from pydub import AudioSegment
from tempfile import mkdtemp


def remove_unwanted_files(directory):
    unwanted_files = [".DS_Store", "desktop.ini"]
    for root, _, files in os.walk(directory):
        for file in files:
            if file in unwanted_files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Eliminato file non desiderato: {file_path}")


def find_duplicate_files_by_name(directory):
    file_names = {}
    duplicates = []

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename in file_names:
                file_names[filename].append(os.path.join(root, filename))
                if filename not in duplicates:
                    duplicates.append(filename)
            else:
                file_names[filename] = [os.path.join(root, filename)]

    return duplicates, file_names


def move_duplicates_to_temp(project_directory, file_paths):
    temp_dir = mkdtemp(dir=project_directory, prefix="temporary")
    for filename, paths in file_paths.items():
        if len(paths) > 1:
            for path in paths:
                rel_path = os.path.relpath(path, start=project_directory)
                dest_path = os.path.join(temp_dir, rel_path)
                dest_dir = os.path.dirname(dest_path)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy2(path, dest_path)
    return temp_dir


def check_audio_duplicates(temp_dir):
    files = os.listdir(temp_dir)
    audio_segments = {}

    for filename in files:
        file_path = os.path.join(temp_dir, filename)
        try:
            audio = AudioSegment.from_file(file_path)
            audio_segments[filename] = audio
        except Exception as e:
            print(f"Errore nel caricamento del file {filename}: {e}")
            continue

    duplicates = []
    checked = set()

    for filename, audio in audio_segments.items():
        if filename in checked:
            continue
        for other_filename, other_audio in audio_segments.items():
            if filename != other_filename and other_filename not in checked:
                if audio == other_audio:
                    duplicates.append((filename, other_filename))
                    checked.add(other_filename)
        checked.add(filename)

    return duplicates


def delete_duplicates(file_names, audio_duplicates):
    deleted_paths = []
    for dup in audio_duplicates:
        for path in file_names[dup[0]]:
            if os.path.exists(path):
                os.remove(path)
                deleted_paths.append(path)
                print(f"Eliminato: {path}")
        for path in file_names[dup[1]]:
            if os.path.exists(path):
                os.remove(path)
                deleted_paths.append(path)
                print(f"Eliminato: {path}")


def main(directory):
    remove_unwanted_files(directory)

    duplicates, file_names = find_duplicate_files_by_name(directory)

    if not duplicates:
        print("Nessun duplicato trovato.")
        return

    print(f"Trovati {len(duplicates)} file duplicati per nome:")
    for filename, paths in file_names.items():
        if len(paths) > 1:
            print(f"{filename}: {len(paths)} volte")

    # Directory del progetto per la cartella temporanea
    project_directory = "C://underwater-classification"
    temp_dir = move_duplicates_to_temp(project_directory, file_names)
    audio_duplicates = check_audio_duplicates(temp_dir)

    if not audio_duplicates:
        print("Nessun duplicato audio trovato.")
    else:
        print(f"Trovati {len(audio_duplicates)} duplicati audio:")
        for dup in audio_duplicates:
            print(f"{dup[0]} e {dup[1]} sono duplicati.")

        delete_duplicates(file_names, audio_duplicates)


if __name__ == "__main__":
    directory = "C://underwater-classification//dataset"
    main(directory)
