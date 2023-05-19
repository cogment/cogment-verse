# Copyright 2022 AI Redefined Inc. <dev+cogment@ai-r.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tarfile
import time
from typing import List, Union

import boto3
from botocore.exceptions import ClientError


def is_valid_source_dir(dir_name: str) -> bool:
    """Ingore pycache and dot folder"""
    return not (dir_name.startswith(".") or dir_name == "__pycache__")


def is_valid_file(file_name: str) -> bool:
    """Ignore dot and cpython files"""
    return not (file_name.startswith(".") or file_name.endswith(".pyc") or file_name.endswith(".pyo"))


def add_files_to_archive(tar: tarfile.TarFile, root: str, files: List[str], project_dir: str) -> None:
    """Adds valid files to a tar archive.

    Args:
        tar (tarfile.TarFile): The tar archive to add files to.
        root (str): The root directory of the files to be added.
        files (List[str]): A list of file names to add to the tar archive.
        project_dir (str): The project directory.

    Returns:
        None
    """
    for file in files:
        if is_valid_file(file):
            path = os.path.join(root, file)
            arcname = os.path.relpath(path, project_dir)
            tar.add(path, arcname=arcname)


def pack_archive(
    project_dir: str,
    main_dir: str,
    output_path: str,
    source_dir_names: Union[List[str], None] = None,
    ignore_folders: Union[List[str], None] = None,
    archive_name: str = "source_code.tar.gz",
):
    """
    Create a compressed archive of the project data.

    Args:
        project_dir (str): Path to the project directory.
        source_dir_names (List[str], optional): List of directory names to include in the archive. Defaults to None.
        ignore_folders (List[str], optional): List of directory names to exclude from the archive. Defaults to None.
        archive_name (str, optional): Name of the compressed archive file. Defaults to "source_code.tar.gz".

    Returns:
        None.

    Raises:
        ValueError: If `source_dir_names` contains a file outside of `project_dir`.
    """
    os.chdir(project_dir)

    if ignore_folders is None:
        ignore_folders = []
    archive_path = os.path.join(output_path, archive_name)
    with tarfile.open(archive_path, "w:gz") as tar:
        if source_dir_names is None:
            for root, dirs, files in os.walk(project_dir):
                dirs[:] = [d for d in dirs if is_valid_source_dir(d) and d not in ignore_folders]
                add_files_to_archive(tar, root, files, project_dir)
        else:
            for source_dir_name in source_dir_names:
                if (
                    os.path.isdir(source_dir_name)
                    and is_valid_source_dir(source_dir_name)
                    and source_dir_name not in ignore_folders
                ):
                    for root, dirs, files in os.walk(source_dir_name):
                        dirs[:] = [d for d in dirs if is_valid_source_dir(d) and d not in ignore_folders]
                        add_files_to_archive(tar, root, files, project_dir)
                elif os.path.isfile(source_dir_name) and is_valid_file(source_dir_name):
                    arcname = os.path.relpath(source_dir_name, project_dir)
                    tar.add(source_dir_name, arcname=arcname)

    os.chdir(main_dir)


def unpack_archive(archive_path: str, source_dir: str):
    """Unpack source code"""
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(source_dir)


def delete_archive(archive_path: str):
    """Remove packed source code"""
    if os.path.exists(archive_path):
        os.remove(archive_path)
    else:
        print(f"The specified archive '{archive_path}' does not exist.")


def upload_to_s3(local_path: str, bucket: str, s3_key: str):
    """Upload a file from local_path to an S3 bucket at the specified key.

    Args:
        local_path (str): Path of the file to upload.
        bucket (str): Name of the S3 bucket to upload to.
        s3_key (str): Key of the object in the S3 bucket.

    Returns:
        None.
    """
    s3_storage = boto3.client("s3")
    with open(local_path, "rb") as file_obj:
        s3_storage.upload_fileobj(file_obj, bucket, s3_key)


def download_from_s3(bucket: str, s3_key: str, local_path: str):
    """Download packed data from S3.

    Args:
        bucket (str): A string representing the S3 bucket name.
        s3_key (str): A string representing the S3 object key.
        local_path (str): A string representing the local path where
            the downloaded file will be saved.

    Returns:
        None
    """
    s3_storage = boto3.client("s3")
    try:
        # Make sure the local directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_storage.download_file(bucket, s3_key, local_path)
    except ClientError as error:
        print(f"Error downloading file from S3: {bucket}/{s3_key}. Details: {error}")


def download_and_extract_data_from_s3(
    bucket: str, s3_key: str, download_path: str, archive_name: str, unpack_path: str
) -> None:
    """Download the data from s3, extract to local path, and delete the packed file"""
    # Download the data
    archive_path = f"{download_path}/{archive_name}"
    download_from_s3(bucket=bucket, s3_key=s3_key, local_path=archive_path)

    # Uppack the data
    unpack_archive(archive_path=archive_path, source_dir=unpack_path)
    time.sleep(0.5)

    # Delete the packed file
    delete_archive(archive_path=archive_path)


def pack_files(input_folder, output_folder, archive_name, ignore_folders: str = None):
    """Simple packing file"""
    if ignore_folders is None:
        ignore_folders = []
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create the output tar file path
    tar_path = os.path.join(output_folder, f"{archive_name}.tar")

    # Create a tar archive
    with tarfile.open(tar_path, "w") as tar:
        for root, dirs, files in os.walk(input_folder):
            dirs[:] = [d for d in dirs if d not in ignore_folders]
            for file in files:
                file_path = os.path.join(root, file)
                tar.add(file_path, arcname=os.path.relpath(file_path, input_folder))


def create_bucket_if_not_exists(bucket_name: str, region: str):
    """Create a s3 bucket if not exisits"""
    s3_storage = boto3.resource("s3")
    try:
        s3_storage.meta.client.head_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' already exists.")
    except ClientError as err:
        error_code = err.response["Error"]["Code"]
        if error_code == "404":
            print(f"Bucket '{bucket_name}' does not exist. Creating the bucket...")
            s3_storage.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": region})
            print(f"Bucket '{bucket_name}' created.")
        else:
            print(f"Error checking bucket '{bucket_name}': {err}")


def aws_creden_setup(profile_name: str) -> None:
    """Setting access credential for AWS"""
    # Profile setup
    session = boto3.Session(profile_name=profile_name)

    # Get the access key and secret key for the specified profile
    credentials = session.get_credentials()
    access_key = credentials.access_key
    secret_key = credentials.secret_key

    # Set the environment variables
    os.environ["AWS_ACCESS_KEY_ID"] = access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key
