# SMOTE
This class is the implementation of SMOTE technique of oversampling the data points in the imbalanced dataset.

## 🗃️ Which files?
In this repository, 2 files are there.
1. `.ipynb` which is the step by step explanation of my approach
2. `.py` which is the standalone SMOTE class for direct use.

## 🤔 How to import?
This is not the library but instead a direct implementation. So where erver you want to use the SMOTE class, you can simply copy the file there and then follow the standard import syntax.

Here I have implemented in the simplest way possible and is dependent
upon only `numpy` and `sklearn's linear regression`.

ℹ️ About
-----

After you detect that you have got the problem of imbalanced instances
per classes, you can simply use this class with ease.

After initializing, SMOTE class's object also has `__repr__` which will
display useful information.


🤓 Parameters
----------

While initializing:

    X: DF / ndarray / list
        It must be 2D and numerical in nature.

    y: Series / array / list
        It must be 1D and should have categories in it.

    oversampling_size: float (between 0 to 1)
        It ensures that how much percentage of highset frequencied class
        you want to generate new samples for. 1 means 100% which means
        total frequences of all class will be same.


While Resampling:

    k: int
        This defines how many neighbours to choose while generating a new
        data point. Default is 5.



👨‍💻 How To
------
```python
# Seperate the Features X and Labels Y
# X should be numerical and 2D.
X = df[["feature1", "feature2"]]
y = df["target"]

synthesizer = SMOTE(oversampling_size=0.5) # means 50% samples generated.
synthesizer.resample(k=3)

# ↓ this will return a dict with new data points
new_data = synthesizer.coords_by_class
```

Thanks!<br>
**Aayush** ∞ **Shah**
