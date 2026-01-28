import json
from pathlib import Path


def stream_json_array(file_path):
    """
    Generator that yields individual objects from a JSON array file
    without loading the entire file into memory.

    Works by tracking brace/bracket depth and string state.
    """
    with open(file_path, "r", encoding="utf-8", buffering=8 * 1024 * 1024) as f:
        char = f.read(1)
        while char and char != "[":
            char = f.read(1)

        if not char:
            return

        buffer = []
        depth = 0
        in_string = False
        escape_next = False

        while True:
            char = f.read(1)
            if not char:
                break

            if escape_next:
                buffer.append(char)
                escape_next = False
                continue

            if char == "\\" and in_string:
                buffer.append(char)
                escape_next = True
                continue

            if char == '"':
                in_string = not in_string
                buffer.append(char)
                continue

            if in_string:
                buffer.append(char)
                continue

            if char == "{" or char == "[":
                if depth == 0 and char == "{":
                    buffer = [char]
                else:
                    buffer.append(char)
                depth += 1
            elif char == "}" or char == "]":
                depth -= 1
                if depth == 0 and char == "}":
                    buffer.append(char)
                    obj_str = "".join(buffer)
                    try:
                        yield json.loads(obj_str)
                    except json.JSONDecodeError:
                        pass
                    buffer = []
                elif depth >= 0:
                    buffer.append(char)
                if depth < 0:
                    break
            elif depth > 0:
                buffer.append(char)


def convert_json_to_jsonl(input_path, output_path):
    """
    Convert a JSON array file to JSONL format using streaming.
    Memory usage stays constant regardless of file size.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    count = 0
    with open(output_path, "w", encoding="utf-8", buffering=8 * 1024 * 1024) as outfile:
        for obj in stream_json_array(input_path):
            outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1

    return count


def main():
    project_root = Path(__file__).parent.parent
    input_dir = project_root / "processed_data"
    output_dir = project_root / "processed_data_jsonl"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Converting JSON to JSONL Format (Streaming)")
    print("=" * 60)

    files = [
        ("val.json", "val.jsonl"),
        ("test.json", "test.jsonl"),
        ("train.json", "train.jsonl"),
    ]

    for json_name, jsonl_name in files:
        input_file = input_dir / json_name
        output_file = output_dir / jsonl_name

        if not input_file.exists():
            continue

        if output_file.exists():
            continue

        convert_json_to_jsonl(input_file, output_file)

    print("\n" + "=" * 60)
    print("Conversion Complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
