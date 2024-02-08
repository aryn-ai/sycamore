# Running Custom Data Preparation Code

If you have chosen to write your data preparation code with an editor, you can run it using the terminal in the Jupyter container or the Sycamore-Importer container. You can also write and run code using a [Jupyter notebook](using_Jupyter.md).

## Using the terminal in the Jupyter container

The easiest way to run your data preparation code is to use the Jupyter notebook shell in the Jupyter container, which is launched by default.

1. Add your data preparation code to the bind directory so you can access it in the Jupyter container. On your local machine, add your files to:

```../jupyter/bind_dir```

2. In your browser, go to the URL for your [Jupyter container](using_jupyter.md) and create a new terminal. You can do this by clicking on "File" in the menu, going to "New," and then selecting "Terminal".

3. In the terminal, run this command in the `/app/work/` directory:

`python bind_dir/your-file-name.py`


## Using the Sycamore-Importer container

You can also copy your code to the Sycamore-Importer container and run it there. However, we don’t recommend this method, and instead we suggest you use the Jupyter methods above. If you do copy your file to the Sycamore-Importer container, we recommend you save it to `/app/.scrapy` so it persists. 

1. Copy your file to the Sycamore-Importer container:


```
docker exec [name-of-your-Sycamore-Importer-Container] 'mkdir sycamore-jobs`
docker cp . [name-of-your-Sycamore-Importer-Container]:/sycamore-jobs
```


2. Run your code:

```
docker exec --workdir /sycamore-jobs [name-of-your-Sycamore-Importer-Container] 'python your-file-name.py'
```



## Running Sycamore locally

You can run Sycamore’s data processing library locally and load the output into your Sycamore stack. Make sure to set your os_client_args in your code to the endpoint of your Sycamore stack, which is port 9200:

```
#example configured to load a Sycamore stack running locally

os_client_args = {
        "hosts": [{"host": "opensearch", "port": 9200}],
        "http_compress": True,
        "http_auth": ("admin", "admin"),
        "use_ssl": False,
        "verify_certs": False,
        "ssl_assert_hostname": False,
        "ssl_show_warn": False,
        "timeout": 120,
    }
```

To run your Sycamore job:

```python /path/to/your-file-name.py```

