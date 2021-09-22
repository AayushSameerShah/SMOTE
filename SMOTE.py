
from sklearn.linear_model import LinearRegression
import numpy as np

class SMOTE:
    """
    This class is the implementation of SMOTE technique of oversampling
    the data points in the imbalanced dataset.
    
    Here I have implemented in the simplest way possible and is dependent
    upon only `numpy` and `sklearn's linear regression`.
    
    About
    -----
    
    After you detect that you have got the problem of imbalanced instances
    per classes, you can simply use this class with ease.
    
    After initializing, SMOTE class's object also has `__repr__` which will
    display useful information.
    
    
    Parameters
    ----------
    
    While initializing:
    ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
        X: DF / ndarray / list
            It must be 2D and numerical in nature.
        
        y: Series / array / list
            It must be 1D and should have categories in it.
            
        oversampling_size: float (between 0 to 1)
            It ensures that how much percentage of highset frequencied class
            you want to generate new samples for. 1 means 100% which means
            total frequences of all class will be same.
    
    
    While Resampling:
    ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
        k: int
            This defines how many neighbours to choose while generating a new
            data point. Default is 5.
    
    
    
    How To
    ------
    
        # Seperate the Features X and Labels Y
        # X should be numerical and 2D.
    >>> X = df[["feature1", "feature2"]]
    ... y = df["target"]
    
    >>> synthesizer = SMOTE(oversampling_size=0.5) # means 50% samples generated.
    >>> synthesizer.resample(k=3)
    
        # ↓ this will return a dict with new data points
    >>> new_data = synthesizer.coords_by_class
    
    
    -- END --
    """
    
    
    def __init__(self, X, y, oversampling_size=1):
        # Simple array transformation
        self.X = np.array(X)
        self.y = np.array(y)
        
        # The length and compatibility checks
        if self.X.ndim != 2:
            raise NotImplementedError("The dimention of X must be 2D.")
        
        if self.X.shape[0] != len(self.y):
            raise NotImplementedError("Length X and Y mismatched.")
        
        if not 0 < oversampling_size <= 1:
            raise NotImplementedError("Oversampling size must be in between 0 to 1.")
        
        # Unique classes and their respective counts
        self.unique_classes, self.unique_count = np.unique(self.y, return_counts=True)
        
        # The max frequency found and multiplied
        # for target frequency generation
        self.to_sample = int(self.unique_count.max() * oversampling_size)
        
        # Total unique classes
        self.nunique = len(self.unique_classes)
        # Total classes which needs oversampling (usually n - 1)
        self.new_generatable = (~(self.unique_count == self.to_sample)).sum()
        
        # Main class with highest freq 
        # (found that 'cause it will be used in next line)
        self.main_class = self.unique_classes[self.unique_count.argmax()]
        # Name of classes to resample (Except the main class)
        self.classes_to_resample = self.unique_classes[self.unique_classes != self.main_class]
        
    
    def resample(self, k=5):
        self.coords_by_class = {}
        # Will iterate through gereratable classes
        for class_ in self.classes_to_resample:
            print(class_)

            # Filter rows to just *that* class
            X_mini = self.X[self.y == class_]
            
            # Fetch class count so we know how much to generate
            class_count = self.unique_count[np.where(self.unique_classes == class_)[0][0]]
            generate_this_many = self.to_sample - class_count

            # Hey, this ↓ should be true.
            assert generate_this_many <= self.to_sample
            # A little check if our "to be generated" class
            # doesn't have less than k instances
            if k >= class_count: k = class_count - 1
                    
            new_coordinates = self.oversample(X_mini, generate_this_many, k)
            self.coords_by_class[class_] = new_coordinates
    
    def oversample(self, X_mini, generate_this_many, k):
        # will have all new generated points `generate_this_many` times
        # and will be returned (per class)
        new_c = []
        
        # will iterate for given times to generate
        # that many.
        for th_sample in range(generate_this_many):
            # One random row
            random_idx = np.random.randint(0, X_mini.shape[0])
            row = X_mini[random_idx]

            # All distances from that row
            distances = ((row - X_mini) ** 2).sum(1) ** 0.5
        
            # `k` neighbours (except itself) 
            sorted_idxes = distances.argsort()[1:k+1]
            k_neighbours = X_mini[sorted_idxes]
            
            # Now picking 1 random neighbor
            rand_neighbour = np.random.randint(0, k)
            neighbour = k_neighbours[rand_neighbour]

            # We have point A's corrs (features)
            # And poinf B's corrs (features)
            B0, Bi = self.get_regression_line(row, neighbour)
            
            # It will handle X1, X2 ... Xn-1 features
            # Xn will be our y (to be predicted)
            new_coordinates = []
            # We will iterate through 2 rows (always 2 
            # as 2 points A and B)
            for p1, p2 in zip(row[:-1], neighbour[:-1]):
                # Get new random coord
                new_coordinates.append(np.random.uniform(p1, p2))
            
            # This will take new coords [X1...Xn-1] along
            # with B0 and Bi to generate the y
            y = self.generator(new_coordinates, B0, Bi)
            
            # Will append in the new coords [X1...Xn-1, Xn]
            new_coordinates.append(y)
            # Appends whole coordinate till `th` times
            new_c.append(new_coordinates)
        return new_c
        
    
    def get_regression_line(self, A, B):
        # Stacking 2 rows for easier access
        data = np.vstack([A, B])
        
        # Make the last column as `y`
        X = data[:, :-1]
        y = data[:, -1]
        
        # Learn the model for 2 rows.
        model = LinearRegression()
        model.fit(X, y)
        B0 = model.intercept_
        Bi = model.coef_
        return B0, Bi
        
        
    @staticmethod
    def generator(new_coords, B0, Bi):
        return B0 + (new_coords * Bi).sum()
    
        
    def __repr__(self):
        return \
        f"""
        Number of features: {self.X.shape[1]}
        Total unique classes: {self.nunique}
        New instances to generate for: {self.new_generatable} class(s)
        Total class balance: {self.to_sample}
        Class with highest instances: {self.main_class}
        """
