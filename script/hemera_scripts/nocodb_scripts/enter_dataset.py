#!/usr/bin/env python3

import argparse
import json
import os
import sys

import prepare_json
import nocodb

"""
Read one ore more 'DATASET.json' files and add it's contents to an existing NocoDB table
"""

if __name__ == '__main__':

    nocodb_host = os.getenv('NOCODB_HOST', 'http://localhost')
    nocodb_token = os.getenv('NOCODB_TOKEN', 'none')
    nocodb_ssl_cert = os.getenv('NOCODB_SSL_CERT', True)

    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs='+', help="Input JSON files")
    parser.add_argument("-t", "--table", required=True, help="Existing Nocodb table id, then insert into this table")
    parser.add_argument("-p", "--prefix", default="",
                        help="Prefix to the URL field which is usually the host part of the URL but can be any prefix string")
    args = parser.parse_args()

    tableid = args.table
    table = nocodb.Table( tableid, host=nocodb_host, auth_token=nocodb_token, verify=nocodb_ssl_cert )

    for f in args.inputs:

        print("    Read ", f)
        with open(f) as ff:

            input = json.load(ff)

            # set special extra fields:
            extra= {}
            extra['URL']= args.prefix + ": " + f 
            extra['status']= "current"
            extra['branch']= "blahblah"

            data = prepare_json.prepare_input( input, extra=extra )

            #print(f"Enter input from {f} into table {args.table}")
            #print(f"        insert: ", json.dumps(data, indent=4) )
            res = table.insert(data)
            #print("    RESULT: ", json.dumps(res, indent=4))
