import os


def extract_paragraphs(directory, output_file):
    with open(output_file, 'w') as outfile:
        for filename in sorted(os.listdir(directory)):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r') as infile:
                    paragraph = []
                    paragraph_count = 1
                    for line in infile:
                        stripped_line = line.strip()
                        if stripped_line:
                            paragraph.append(stripped_line)
                        elif paragraph:
                            prefix = f"[{filename}_{paragraph_count:02d}] "
                            outfile.write(prefix + ' '
                                          .join(paragraph) + '\n\n')
                            paragraph = []
                            paragraph_count += 1
                    if paragraph:
                        prefix = f"[{filename}_{paragraph_count:02d}] "
                        outfile.write(prefix + ' '.join(paragraph) + '\n\n')


if __name__ == "__main__":
    # Ensure these paths exist on your machine
    directory = "./input_texts"
    # Changed from a directory-style path to a specific file path
    output_file = "./output_paragraphs/extracted_results.txt"
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    extract_paragraphs(directory, output_file)
    print(f"Paragraphs have been extracted to {output_file}")
