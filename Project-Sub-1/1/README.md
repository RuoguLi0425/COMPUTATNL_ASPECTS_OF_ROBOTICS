# Getting Started
Make sure you are running python3. Dependencies for hw1 can be installed using the following command:

```
python -m venv hw1
source hw1/bin/activate
pip install -r requirements.txt
```

This will create a virtual environment where the necessary dependencies will be accessible.
You can read more about python virual environments [here](https://pymbook.readthedocs.io/en/latest/virtualenv.html).


# Helpful Scripts and Tips
Please refer to the homework handout for details.

Once you have filled out `transforms.py` you can run the following to test your functions:
```bash
python transforms_test.py
```
It is highly recomended that all of these tests pass before you proceed.

Once you have implemented `ply.py`, try instantiating a `Ply` object. Make sure that you can both read in and write out `data/point_sample.ply` and `data/triangle_sample.ply`.

Now you are ready to implement the RGB-D frame fusion functions in `tsdf.py`.

After implementing `tsdf.py` you can call your fusion on the sample data provided by running the following:
```bash
python tsdf_run.py
```
Compare these with the `data/*.png` images to make sure the output `mesh.ply` and `point_cloud.ply` from your tsdf code looks good!

Please document any known bugs in your code or other notes that you would like TAs to consider.