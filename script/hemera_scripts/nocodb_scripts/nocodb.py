# nocodb class

from enum import Enum

import requests
import json

"""
Class for access to a NOCODB instance
"""


class Nocodb:

    def __init__(self, host, auth_token, verify=True):
        self.host = host
        self.auth_token = auth_token
        self.verify = verify

        self.session = requests.Session()
        self.session.headers.update({"accept": "application/json", "xc-token": self.auth_token})
        self.session.verify = self.verify

    """ 
    [META] List Bases
    List all base meta data
    https://meta-apis-v2.nocodb.com/#tag/Base/operation/base-list
    GET http://localhost:8080/api/v2/meta/bases/
    """

    def list_bases(self) -> dict:

        url = f"{self.host}/api/v2/meta/bases/"

        res = self.session.get(url, params={}).json()

        # print(json.dumps(res, indent=4))

        ret = {}
        for i in res['list']:
            # print( i['title'], i['id'] )
            ret[i['id']] = i['title']

        return ret

    """
    [META] Create Base
    Create a new base
    https://meta-apis-v2.nocodb.com/#tag/Base/operation/base-create
    POST http://localhost:8080/api/v2/meta/bases/
    """

    def create_base(self, title, color=None, description=None, base_type=None) -> dict:

        url = f"{self.host}/api/v2/meta/bases/"

        data = {"title": title}
        if color:
            data["color"] = color
        if description:
            data["description"] = description
        # TODO this won't be returned
        if base_type:
            data["type"] = base_type.name

        res = self.session.post(url, data=data).json()

        if 'msg' in res and res['msg'] == 'Invalid token':
            raise Exception("Invalid token")

        # print( json.dumps(res, indent=4 ) )

        return res

    def get_base_object(self, title):

        # check if such a table exists
        all_bases = self.list_bases()

        for base_id, base_title in all_bases.items():
            if title == base_title:
                return Base(base_id, nocodb=self)
        return None

    """
    [META] Delete Base
    Delete the given base
    https://meta-apis-v2.nocodb.com/#tag/Base/operation/base-delete
    DELETE http://localhost:8080/api/v2/meta/bases/{baseId}
    """

    def delete_base(self, base_id) -> dict:
        url = f'{self.host}/api/v2/meta/bases/{base_id}'
        headers = {"accept": "application/json", "xc-token": self.auth_token}
        return requests.delete(url, headers=headers).json()


"""
Class for access to a NOCODB base == one database in the Nocodb instance
"""


class Base:

    def __init__(self, base_id, nocodb=None, host=None, auth_token=None, verify=True):

        # Don't keep the parent Nocodb object because we can work without it
        # Accessing a Base only needs the baseid and the auth_token
        # Thus like this one can create a Base object without a Nocodb object.
        # #self.nocodb= nocodb
        self.base_id = base_id
        self.verify = verify
        self.session = None

        if nocodb:

            self.host = nocodb.host
            self.auth_token = nocodb.auth_token
            self.verify = nocodb.verify
            self.session = nocodb.session  # re-use the same session

        else:

            assert None != host
            assert None != auth_token

            self.host = host
            self.auth_token = auth_token

            self.session = requests.Session()
            self.session.headers.update({"accept": "application/json", "xc-token": self.auth_token})
            self.session.verify = self.verify

        self.info = self.get_base_info()
        self.metadata = self.get_base()

    class Type(Enum):
        database = 'database'
        documentation = 'documentation'
        dashboard = 'dashboard'

    def debug(self):
        print("class Base: ", self.base_id)
        print(json.dumps(self.info, indent=4))
        print(json.dumps(self.metadata, indent=4))

    """
    [META] Get Base info
    Get info such as node version, arch, platform, is docker, rootdb and package version of a given base
    https://meta-apis-v2.nocodb.com/#tag/Base/operation/base-meta-get
    GET http://localhost:8080/api/v2/meta/bases/{baseId}/info
    """

    def get_base_info(self) -> dict:

        url = f"{self.host}/api/v2/meta/bases/{self.base_id}/info"

        res = self.session.get(url, params={}).json()

        # print( json.dumps(res, indent=4 ) )

        return res

    """
    [META] Get Base
    Get the info of a given base
    https://meta-apis-v2.nocodb.com/#tag/Base/operation/base-read
    GET http://localhost:8080/api/v2/meta/bases/{baseId}
    """

    def get_base(self) -> dict:

        url = f"{self.host}/api/v2/meta/bases/{self.base_id}"

        res = self.session.get(url, params={}).json()

        # print( json.dumps(res, indent=4 ) )

        return res

    """
    [META] List Tables
    List all tables in a given base
    https://meta-apis-v2.nocodb.com/#tag/DB-Table/operation/db-table-list
    GET http://localhost:8080/api/v2/meta/bases/{baseId}/tables
    """

    def list_tables(self) -> dict:

        url = f"{self.host}/api/v2/meta/bases/{self.base_id}/tables"

        res = self.session.get(url, params={}).json()

        # print( json.dumps(res, indent=4 ) )

        ret = {}
        for i in res['list']:
            # print( i['title'], i['id'] )
            ret[i['title']] = i['id']

        return ret

    """
    [META] Create Table
    Create a new table in a given base
    https://meta-apis-v2.nocodb.com/#tag/DB-Table/operation/db-table-create
    POST http://localhost:8080/api/v2/meta/bases/{baseId}/tables
    TBD
    """

    def create_table(self, name, columns, color=None):

        url = f"{self.host}/api/v2/meta/bases/{self.base_id}/tables"

        data = {
            "columns": columns,
            "table_name": name,
            "title": name
        }
        # print( json.dumps(data, indent=4 ) )

        if color:
            data["color"] = color

        res = self.session.post(url, json=data).json()

        if "errors" in res or "msg" in res:
            print("ERROR: ")
            print(json.dumps(res, indent=4))
            return None

        # print( json.dumps(res, indent=4 ) )
        # print(" --> newly created table id ", res['id'] )

        return res['id']

    def get_table_object(self, title):

        # check if such a table exists
        alltables = self.list_tables()

        if title in alltables:

            return Table(alltables[title], base=self)

        else:

            return None


"""
Class for access to a NOCODB table
"""


class Table:

    def __init__(self, tableid, base=None, host=None, auth_token=None, verify=True):

        # Don't keep the parent Base object because we can work without it
        # Accessing a Table only needs the tableid and the auth_token
        # Thus like this one can create a Table object without a Base object.
        # self.base= base
        self.tableid = tableid
        self.verify = verify
        self.session = None

        if base:

            self.host = base.host
            self.auth_token = base.auth_token
            self.verify = base.verify
            self.session = base.session

        else:

            assert None != host
            assert None != auth_token

            self.host = host
            self.auth_token = auth_token

            self.session = requests.Session()
            self.session.headers.update({"accept": "application/json", "xc-token": self.auth_token})
            self.session.verify = self.verify

        # connection test
        res = self.count()
        assert 'count' in res
        assert 'msg' not in res

    def debug(self):
        # print("class Table: ", self.tableid )
        metadata = self.read_table()
        metadata['columnsById'] = "<skipped>"
        print(json.dumps(metadata, indent=4))

    """
    [META] Read Table
    Read the table meta data by the given table ID
    https://meta-apis-v2.nocodb.com/#tag/DB-Table/operation/db-table-read
    GET http://localhost:8080/api/v2/meta/tables/{tableId}
    """

    def read_table(self):

        url = f"{self.host}/api/v2/meta/tables/{self.tableid}"

        res = self.session.get(url, params={}).json()

        # print( json.dumps(res, indent=4 ) )

        return res

    """
    [DATA] Create Table Records
    https://data-apis-v2.nocodb.com/#tag/Table-Records/operation/db-data-table-row-list
    POST http://localhost:8080/api/v2/tables/{tableId}/records
    """

    def insert(self, data) -> dict:

        url = f"{self.host}/api/v2/tables/{self.tableid}/records"

        # print( json.dumps(data, indent=4 ) )

        res = self.session.post(url, json=data).json()

        # print( json.dumps(res, indent=4 ) )

        return res

    """
    [DATA] Update Table Records
    https://data-apis-v2.nocodb.com/#tag/Table-Records/operation/db-data-table-row-update
    PATCH http://localhost:8080/api/v2/tables/{tableId}/records
    """

    def update(self, data):

        # print("AA")
        # print("TABLE DELETE ", json.dumps(list,indent=4) )
        # print("    JSON ", json.dumps(list))

        url = f"{self.host}/api/v2/tables/{self.tableid}/records"

        # print( json.dumps(data, indent=4 ) )
        res= self.session.patch(url, data=data).json()

        return res

    """
    [DATA] Read Table Record
    This API endpoint allows you to retrieve a single record identified by Record-ID, serving as unique identifier for the record from a specified table.
    https://data-apis-v2.nocodb.com/#tag/Table-Records/operation/db-data-table-row-read
    GET http://localhost:8080/api/v2/tables/{tableId}/records/{recordId}
    """

    def read_table_record(self, primary_key: any) -> dict:
        url = f"{self.host}/api/v2/tables/{self.tableid}/records/{primary_key}"
        return self.session.get(url).json()

    def update_record(self, pk: int, key: str, value: object) -> dict:
        # record = self.read_table_record(pk)
        # if 'CreatedAt' in record:
        #     del record['CreatedAt']
        # if 'UpdatedAt' in record:
        #     del record['UpdatedAt']
        # if key == 'Primary Key':
        #     raise ValueError("Primary Key is not overridable")
        # record[key] = value
        record = [{'Primary Key': pk, f'{key}': value}]
        return self.update(record)

    """ 
    [DATA] Count Table Records
    This API endpoint allows you to retrieve the total number of records from a specified table or a view. You can narrow down search results by applying where query parameter
    https://data-apis-v2.nocodb.com/#tag/Table-Records/operation/db-data-table-row-count
    Get http://localhost:8080/api/v2/tables/{tableId}/records/count
    """

    def count(self, where=None):

        url = f"{self.host}/api/v2/tables/{self.tableid}/records/count"

        params = {}

        if where:
            params['where'] = where

        res = self.session.get(url, params=params).json()

        return res

    """ 
    [DATA] List Table Records
    This API endpoint allows you to retrieve records from a specified table. 
    https://data-apis-v2.nocodb.com/#tag/Table-Records/operation/db-data-table-row-list
    Get http://localhost:8080/api/v2/tables/{tableId}/records
    """

    def list(self, fields=None, sort=None, where=None, offset=0, limit=None):

        url = f"{self.host}/api/v2/tables/{self.tableid}/records"

        params = {}

        if fields:
            params['fields'] = fields
        if sort:
            params['sort'] = sort
        if where:
            params['where'] = where
            print("                WHERE ", where)
        if limit:
            params['limit'] = limit

        res = self.session.get(url, params=params).json()

        # print(" #### page info #### ")
        # print( json.dumps(res['pageInfo'], indent=4 ) )
        # print(" ######## ")

        return res


"""
[DATA] Delete Table Records
This API endpoint allows deleting existing records within a specified table identified by 
an array of Record-IDs, serving as unique identifier for the record. Records to be deleted are input as an array of record-identifiers.
https://data-apis-v2.nocodb.com/#tag/Table-Records/operation/db-data-table-row-delete
DELETE http://localhost:8080/api/v2/tables/{tableId}/records
"""


def delete(self, list=None):
    url = f"{self.host}/api/v2/tables/{self.tableid}/records"

    # print("TABLE DELETE ", json.dumps(list,indent=4) )
    # print("    JSON ", json.dumps(list))

    if list is None:
        res = self.session.delete(url)
    else:
        res = self.session.delete(url, data=list)

    return res
