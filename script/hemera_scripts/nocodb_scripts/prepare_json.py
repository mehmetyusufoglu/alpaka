import sys
import json

import pandas
import dateutil.parser
from numpy.ma.core import append

"""
Prepare JSON metadata from a 'DATASET.json' file before it can be entered to Nocodb
"""

""" 
Helper function to check if a string contains time duration. Test this before is_datetime() because that 
may also answer something but will use the currect day with the time duration as point in time.
"""
def is_time_duration(string):
    try:
        # use pandas datetime parser because it also handles nanoseconds
        result = pandas.to_datetime(string, format='%H:%M:%S.%f', errors='raise')
        return pandas.Timedelta(days=0,
                                hours=result.hour, minutes=result.minute, seconds=result.second,
                                microseconds=result.microsecond, nanoseconds=result.nanosecond)

    except ValueError:
        pass

    try:
        # same but with no second decimals
        # use pandas datetime parser because it also handles nanoseconds
        result = pandas.to_datetime(string, format='%H:%M:%S', errors='raise')
        return pandas.Timedelta(days=0,
                                hours=result.hour, minutes=result.minute, seconds=result.second,
                                microseconds=result.microsecond, nanoseconds=result.nanosecond)

    except ValueError:
        pass

    return None


""" 
Helper function to check if a string contains datetime
"""
def is_datetime(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try:
        result = dateutil.parser.parse(string, fuzzy=fuzzy)
        return result

    except ValueError:
        return None


"""
Recursively flatten the keys in a nested JSON structure
- store resulting keys in 'result_keys' and resulting flat JSON in 'result_data'
- JSON dicts are flattened
- JSON arrays are kept like they are
- text and numbers are copied
- use '.' as separators for flattened keys
    - check if there are collisions between existing names and flattened names
"""


def flatten_keys(result, prefix, json_data, separator="."):
    # only append a separator if prefix was not empty
    if "" != prefix:
        prefix = prefix + separator

    for k in json_data:

        # replace spaces in names by underscore
        kk = k.replace(' ', '_')

        # if kk != k:
        #     print("    rename ",k, " --> ", kk, " as ", type( json_data[k] ))

        if type({}) == type(json_data[k]):

            # print("        dict")

            # recursive
            flatten_keys(result, prefix + kk, json_data[k])

        else:

            flat_key = prefix + kk
            if flat_key in result:
                print("WARNING: ", flat_key, " already present")

            result[flat_key] = json_data[k]


def prepare_input(input, extra=None):
    # print( json.dumps(input,indent=4) )

    data = {}

    if extra:
        data = extra
        # print("extra keys ",extra)

    flatten_keys(data, "", input)
    fix_datatypes(data)

    # print("Result: ")
    # print( json.dumps(data,indent=4))

    return data

# DEPRECATED
def append_column_metadata(columns, k, v):
    if type(v) is int:

        # print("    ", k, " --> type ", type(v), " example ", v)
        # print("handle int")
        columns.append(
            {
                "column_name": k,
                "title": k,
                "uidt": "Number",
                "dt": "int",
                "un": False
            })

    elif type(v) is float:

        # print("    ", k, " --> type ", type(v), " example ", v)
        # print("handle float or decimal")
        columns.append(
            {
                "column_name": k,
                "title": k,
                "uidt": "Number",
                "dt": "float8",
                "un": False
            })

    elif type(v) is str:

        # print("    ", k, " --> type ", type(v), " example ", v)

        # several different cases here

        if time_duration := is_time_duration(v):

            # print("handle string -- duration")
            # print("    found duration ", time_duration, " from ", v)

            columns.append(
                {
                    "column_name": k,
                    "title": k,
                    "uidt": "Duration",
                    "dt": "decimal",
                    "un": False
                })

        elif datetime := is_datetime(v):

            # print("handle string -- datetime")
            # print("    found datetime ", datetime, " from ", v)
            columns.append(
                {
                    "column_name": k,
                    "title": k,
                    "uidt": "DateTime",
                    "dt": "timestamp",
                    "un": False
                })

        elif v.startswith("http://") or v.startswith("https://"):

            # print("handle string -- url")
            # print("    found datetime ", datetime, " from ", v)
            columns.append(
                {
                    "column_name": k,
                    "title": k,
                    "uidt": "URL",
                    "dt": "character varying",
                    "un": False
                })
        else:

            # print("handle string -- generic")
            columns.append(
                {
                    "column_name": k,
                    "title": k,
                    "uidt": "SingleLineText",
                    "dt": "character varying",
                    "un": False
                })


    elif type(v) is bool:

        # print("    ", k, " --> type ", type(v), " example ", v)
        # print("handle bool")
        columns.append(
            {
                "column_name": k,
                "title": k,
                "uidt": "Checkbox",
                "dt": "bool",
                "un": False
            })

    elif type(v) is list:

        # print("    ", k, " --> type ", type(v), " example ", v)
        # print("handle list")
        columns.append(
            {
                "column_name": k,
                "title": k,
                "uidt": "JSON",
                "dt": "json",
                "un": False
            })

    elif v is None:

        # special case NoneType -- handle as string

        # print("handle NoneType as string")
        columns.append(
            {
                "column_name": k,
                "title": k,
                "uidt": "SingleLineText",
                "dt": "character varying",
                "un": False
            })

    else:

        ## no not cover {} above because those should have been resolved by 'flatten_keys()'

        print("    ", k, " --> type ", type(v), " example ", v)
        print("handle unknown, abort")
        sys.exit(-2)

def add_column_definition( columns, k, type, subtype="" ):

    if "integer" == type :

        # print("    ", k, " --> type ", type(v), " example ", v)
        # print("handle int")
        columns.append(
            {
                "column_name": k,
                "title": k,
                "uidt": "Number",
                "dt": "int",
                "un": False
            })

    elif "number" == type: # float

        # print("    ", k, " --> type ", type(v), " example ", v)
        # print("handle float or decimal")
        columns.append(
            {
                "column_name": k,
                "title": k,
                "uidt": "Number",
                "dt": "float8",
                "un": False
            })

    elif "string" == type:

        # print("    ", k, " --> type ", type(v), " example ", v)

        # several different cases here

        if "Duration" == subtype:

            # print("handle string -- duration")
            # print("    found duration ", time_duration, " from ", v)

            columns.append(
                {
                    "column_name": k,
                    "title": k,
                    "uidt": "Duration",
                    "dt": "decimal",
                    "un": False
                })

        elif "DateTime" == subtype:

            # print("handle string -- datetime")
            # print("    found datetime ", datetime, " from ", v)
            columns.append(
                {
                    "column_name": k,
                    "title": k,
                    "uidt": "DateTime",
                    "dt": "timestamp",
                    "un": False
                })

        elif "URL" == subtype:

            # print("handle string -- url")
            # print("    found datetime ", datetime, " from ", v)
            columns.append(
                {
                    "column_name": k,
                    "title": k,
                    "uidt": "URL",
                    "dt": "character varying",
                    "un": False
                })
        else:

            # print("handle string -- generic")
            columns.append(
                {
                    "column_name": k,
                    "title": k,
                    "uidt": "SingleLineText",
                    "dt": "character varying",
                    "un": False
                })

    elif "boolean" == type:

        # print("    ", k, " --> type ", type(v), " example ", v)
        # print("handle bool")
        columns.append(
            {
                "column_name": k,
                "title": k,
                "uidt": "Checkbox",
                "dt": "bool",
                "un": False
            })

    elif "array" == type:

        # print("    ", k, " --> type ", type(v), " example ", v)
        # print("handle list")
        columns.append(
            {
                "column_name": k,
                "title": k,
                "uidt": "JSON",
                "dt": "json",
                "un": False
            })

    elif "null" == type:

        # special case NoneType -- handle as string

        # print("handle NoneType as string")
        columns.append(
            {
                "column_name": k,
                "title": k,
                "uidt": "SingleLineText",
                "dt": "character varying",
                "un": False
            })

    else:

        ## no not cover {} above because those should have been resolved by 'flatten_keys()'

        print("    ", k, " --> type ", type)
        print("handle unknown, abort")
        sys.exit(-2)


def derive_table_columns_definition(data):
    columns = []

    # column_template= {
    # "column_name": "title",
    # "title": "Title",

    # "uidt": "SingleLineText",
    # "dt": "varchar",

    # "uidt": "Number" ,
    # "dt": "float8" | "int" | "bigint"

    # "uidt": "Decimal" ,
    # "dt": "decimal"

    # "uidt": "DateTime",
    # "dt": "timestamp",

    # "uidt": "Checkbox",
    # "dt": "bool",

    # "uidt": "ID",
    # "dt": "int4",

    # uidt": "SingleSelect",
    # "dt": "text"

    # "un": false
    # }

    # column_template= {
    #     "column_name": 
    #     "title": 
    #     "uidt": 
    #     "dt": 
    #     "un": false
    # }

    for k in data:
        v = data[k]
        append_column_metadata(columns, k, v)

    return columns

"""
Transform some data types -- compare to derive_table_columns_definition()
"""


def fix_datatypes(data):
    for k in data:

        if type("") == type(data[k]):

            if is_it_time_duration := is_time_duration(data[k]):

                # print("fix string -- duration")
                # print("    found duration ", is_it_time_duration, " from ", data[k])

                data[k] = is_it_time_duration.total_seconds()

            elif is_it_datetime := is_datetime(data[k]):

                # print("fix string -- datetime")
                # print("    found datetime ", is_it_datetime, " from ", data[k])

                data[k] = is_it_datetime.isoformat()

## TODO declare primary key
