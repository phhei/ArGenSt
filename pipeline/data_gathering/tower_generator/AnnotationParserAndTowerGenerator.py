import csv
import math
import os

import ezodf
import pandas as pd

path_to_directory = os.path.abspath(os.path.join('..', '..', 'data', 'annotations.ods'))


def parse_annotations_and_write_towers():

    map_generalDimension_mapsOfSingleDimensions = {}

    for i in range(0, 18):
        dataframe = read_ods(i)
        header = generate_header(dataframe)
        map_singleDimension_values = getMeanValuesForEachDimensionAndRow(dataframe, header)

        if header[1] not in map_generalDimension_mapsOfSingleDimensions:
            map_generalDimension_mapsOfSingleDimensions[header[1]] = {}
        map_generalDimension_mapsOfSingleDimensions[header[1]][header[2]] = map_singleDimension_values

    for generalDimension in map_generalDimension_mapsOfSingleDimensions:
        mapOfSingleDimensions = map_generalDimension_mapsOfSingleDimensions[generalDimension]

        columns = []

        # dimension names
        entries = []
        for singleDimension in mapOfSingleDimensions:
            for entity in mapOfSingleDimensions[singleDimension]:
                entries.append(entity)
            break
        columns.append(entries)

        # sub dimensions with values
        for singleDimension in mapOfSingleDimensions:
            map_singleDimension_values = mapOfSingleDimensions[singleDimension]
            entries = []
            for general_dimension in map_singleDimension_values:
                entries.append(map_singleDimension_values[general_dimension])
            columns.append(entries)

        rows = [list(x) for x in zip(*columns)]

        path_to_towers = os.path.abspath(os.path.join('..', 'StanceClassifier', 'submodules', 'towers', 'LinearUserProfileEncoderEmbeddings', str(generalDimension) + '.tsv'))

        with open(path_to_towers, 'w', newline='\n') as csvfile:
            tsv_writer = csv.writer(csvfile, delimiter='\t', quotechar='"')

            # first line with comment
            tsv_writer.writerow(['# column number 1 represents the possible answers'])
            count = 0
            for key in mapOfSingleDimensions:
                tsv_writer.writerow(['# column number ' + str(count + 2) + ' represents the following: ' + str(key[0:-2])])
                count += 1

            for row in rows:
                if type(row[1]) == float and not math.isnan(row[1]):
                    tsv_writer.writerow(row)

            default_row = ['default']
            for _ in mapOfSingleDimensions:
                default_row.append(0)
            tsv_writer.writerow(default_row)

            csvfile.flush()
            csvfile.close()


def getMeanValuesForEachDimensionAndRow(dataframe, header):
    map_singleDimension_values = {}
    for index, row in dataframe.iterrows():
        map_singleDimension_values[row[header[1]]] = mean_of_available_ratings(
            [row[header[2]],
             row[header[3]],
             row[header[4]],
             row[header[5]]
             ],
            4
        )
    return map_singleDimension_values


def generate_header(dataframe):
    header = []
    for column in dataframe:
        header.append(column)
    return header


def read_ods(sheet_no, header=0):
    tab = ezodf.opendoc(filename=path_to_directory).sheets[sheet_no]
    return pd.DataFrame({col[header].value: [x.value for x in col[header + 1:]] for col in tab.columns()})


def mean_of_available_ratings(values, round_digits):
    sum = 0
    count = 0
    for v in values:
        try:
            sum += float(v)
            count += 1
        except Exception:
            continue

    try:
        return round(sum / count, round_digits)
    except:
        return None
