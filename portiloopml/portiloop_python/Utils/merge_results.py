import json
import os
import argparse


def merge_results(directory):
    final_dict = {}
    # Go through each file in the given directory
    for filename in os.listdir(directory):
        # Open the file and read it as a json file
        with open(directory + '/' + filename, 'r') as f:
            print("Loading file: " + filename + "...")
            data = json.load(f)

        for experiment, subject in data.items():
            # If the experiment is not in the dictionary, add it
            if experiment not in final_dict:
                final_dict[experiment] = subject
            else:
                # If the experiment is in the dictionary, add the subject
                final_dict[experiment].update(subject)

    # Write the merged results to a file
    with open('merged_results.json', 'w') as f:
        json.dump(final_dict, f, indent=4)


if __name__ == '__main__':
    # Get command line parameters for the directory
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='Directory containing the results')
    args = parser.parse_args()
    merge_results(args.directory)
