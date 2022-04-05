#!usr/bin/env python3

# Roel Glas
# 26-03-2018
# Fixes by Olga on 13-05-2019
# A script to check whether your uploaded predictions file is in the right format. It check the amount of lines, the amount of columns and the header of the file.

# Import sys module
import sys

# Constants
LINES_NEEDED = 4 #When you run it on your final file for submission, be sure you change this number to 58!
HEADER = "\"Sample\"\t\"Subgroup\""
NO_COLS = 2


# Get filename from arguments
def get_filename():
    try:
        sys.argv[1]
    except:
        sys.exit('\nERROR: No file given\n')

    return(sys.argv[1])


# Load prediction file
def load_linesplit_predictions(filename):
    # Load file and split in lines
    try:
        prediction_object = open(filename,'r')
    except:
        sys.exit("\nERROR: Something went wrong when trying to load the file. Is the filename / path correct?\n")
    prediction_text = prediction_object.read()
    pred_linesplit = prediction_text.splitlines()
    return(pred_linesplit)


# Determine if the number of predictions is correct
def determine_correct_lines(pred_linesplit):
    correct_lines = len(pred_linesplit) == LINES_NEEDED

    if correct_lines:
        correct_lines_string = 'Correct'
    else: 
        correct_lines_string = 'INCORRECT'
    
    return(correct_lines_string)


# Determine if header is correct
def determine_correct_header(pred_linesplit):
    header = pred_linesplit[0]
    correct_header = header == HEADER
    
    if correct_header:
        correct_header_string = 'Correct'
    else:
        correct_header_string = 'INCORRECT'

    return(correct_header_string)


# Determine if amount of columns is correct
def determine_correct_cols(pred_linesplit):
    correct_cols = True
    for i in range(len(pred_linesplit)):
        pred_colsplit = pred_linesplit[i].split('\t')
        correct_cols_for_line = len(pred_colsplit) == NO_COLS
        if correct_cols_for_line == False:
            correct_cols = False

    if correct_cols:
        correct_cols_string = 'Correct'
    else:
        correct_cols_string = 'INCORRECT'
    
    return(correct_cols_string)


# A command line print statement if correct results are obtained
def print_statement(correct_lines_string,
        correct_cols_string,
        correct_header_string):


    print("The amount of lines in you file are: " +
            correct_lines_string + "\n" +
            "The amount of columns in your file are:"+
            correct_cols_string + "\n" +
            "The header in your file is: " +
            correct_header_string)

# Main function
def main():
    filename = get_filename()
    pred_linesplit = load_linesplit_predictions(filename)
    correct_lines_string = determine_correct_lines(pred_linesplit)
    correct_cols_string = determine_correct_cols(pred_linesplit)
    correct_header_string = determine_correct_header(pred_linesplit)
    print_statement(correct_lines_string,
            correct_cols_string,
            correct_header_string)

if __name__ == '__main__':
    main()






