# Running Custom Data Preparation Code

Once you have written your data preparation code, you can now use it to process data when loading it into Sycamore.

## Using Jupyter container

The easiest way to use your data processing code is to use the Jupyter notebook shell in the Sycamore-Jupyter container. You enable this container when deploying Sycamore. [LINK]

Add your code to the bind-dir XXX NEED THIS INFO!

Go to your Jupyter notebook server and go to Jupyter shell

Run this command:

XXXXXXXXXXXXX

You can also choose to run your data processing code from the Docker terminal by:


XXXXXXXXX


## Using Sycamore-Importer container

If you do not choose to run the Jupyter container, you can copy your code to the Sycamore-Importer container and run it there. However, we don’t recommend this, as there is no bind-dir configured for this container.

To run your code:


Copy your code to the Sycamore-Importer container:



XXXXXX



Run the code using this command:


## Running Sycamore locally

You can run Sycamore’s data processing library locally and load the output into your Sycamore stack. To do this, run:



XXXXXXXXXXX



Note: Make sure your code has the proper configuration for your Sycamore stack’s endpoint.



You can also use Jupyter locally for this, and an example is here. [NEED LINK]
