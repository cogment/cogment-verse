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


def package_source_code(
    project_dir: str,
    source_dir_names: Union[List[str], None],
    ignore_folders: Union[List[str], None] = None,
    archive_name: str = "source_code.tar.gz",
):
    """
    Create a compressed archive of the project source code.

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

    with tarfile.open(archive_name, "w:gz") as tar:
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


def unpack_archive(source_archive, source_dir):
    """Unpack source code"""
    with tarfile.open(source_archive, "r:gz") as tar:
        tar.extractall(source_dir)


def delete_archive(archive_path: str):
    """Remove packed source code"""
    if os.path.exists(archive_path):
        os.remove(archive_path)
    else:
        print(f"The specified archive '{archive_path}' does not exist.")


def pack_folder(folder_path: str, archive_name: str, ignore_list: Union[List[str], None]=None):
    """Create a tar archive of a folder, excluding files and directories in the ignore_list.

    Args:
        folder_path (str): Path of the folder to archive.
        archive_name (str): Name of the archive file to create.
        ignore_list (Union[List[str], None]): An optional list specifying files and directories
          to exclude from the archive. Default is None.

    Returns:
        None.
    """
    if ignore_list is None:
        ignore_list = []

    with tarfile.open(archive_name, "w:gz") as tar:
        for root, dirs, files in os.walk(folder_path):
            # Ignore directories in the ignore_list
            dirs[:] = [d for d in dirs if os.path.join(root, d) not in ignore_list]

            for file in files:
                file_path = os.path.join(root, file)

                # Ignore files in the ignore_list
                if file_path in ignore_list:
                    continue

                arcname = os.path.relpath(file_path, folder_path)
                tar.add(file_path, arcname=arcname)

def upload_to_s3(local_path: str, bucket: str, s3_key: str):
    """Upload a file from local_path to an S3 bucket at the specified key.

    Args:
        local_path (str): Path of the file to upload.
        bucket (str): Name of the S3 bucket to upload to.
        s3_key (str): Key of the object in the S3 bucket.

    Returns:
        None.
    """
    s3 = boto3.client("s3")
    with open(local_path, "rb") as file_obj:
        s3.upload_fileobj(file_obj, bucket, s3_key)
    print(f"Uploaded {local_path} to s3://{bucket}/{s3_key}")


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
    s3 = boto3.client("s3")
    with open(local_path, "wb") as file_obj:
        s3.download_fileobj(bucket, s3_key, file_obj)
    print(f"Downloaded s3://{bucket}/{s3_key} to {local_path}")


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
    unpack_archive(source_archive=f"{project_dir}/input/data/training/source_code.tar.gz", source_dir=f"{project_dir}/test_unpack")
