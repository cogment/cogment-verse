from typing import Union, List
import os
import tarfile
import boto3


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
        s3_storage.download_file(bucket, s3_key, local_path)
        print(f"File downloaded successfully from S3: {bucket}/{s3_key}")
    except Exception as error:
        print(f"Error downloading file from S3: {bucket}/{s3_key}")
        print(error)
    print(f"Downloaded s3://{bucket}/{s3_key} to {local_path}")


def download_and_extract_data_from_s3(bucket: str, s3_key: str, local_path: str) -> None:
    """Download the data from s3, extract to local path, and delete the packed file"""
    # Download the data
    download_from_s3(bucket=bucket, s3_key=s3_key, local_path=local_path)

    # Uppack the data
    unpack_archive(archive_path=local_path, source_dir=local_path)

    # Delete the packed file
    delete_archive(archive_path=local_path)


if __name__ == "__main__":
    project_dir = os.getcwd()
    source_dir_names = [
        "actors",
        "cogment_verse",
        "config",
        "environments",
        "runs",
        "tests",
        "main.py",
        "simple_mlflow.py",
    ]
    ignore_folders = ["node_modules"]
    # package_source_code(project_dir=project_dir, source_dir_names=source_dir_names, ignore_folders=ignore_folders)
    source_code_arv = f"{project_dir}/input/data/training/source_code.tar.gz"
    mlflow_arv = f"{project_dir}/.congment_verse/mlflow/"
    pack_archive(
        project_dir=f"{project_dir}/.cogment_verse", source_dir_names=["mlflow"], archive_name="mlflow_db.tar.gz"
    )
    # unpack_archive(archive_path=f"{project_dir}/test_unpack/mlflow_db.tar.gz", source_dir=f"{project_dir}/test_unpack")
