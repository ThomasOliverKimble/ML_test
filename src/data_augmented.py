import numpy as np

def sin_cos_generation(data, col_nbr):
    """Add extra features to data: sinus of column, and cosinus of column.
    Args:
        data: data, shape=(N, D)
        col_nbr: list of columns on which we apply sinus and cosinus, length L
    Returns:
        data: augmented data, shape shape=(N, D+2*L)
    """
    for col in col_nbr:
        sin_col = np.sin(data[:,col])
        sin_col = sin_col[:, np.newaxis]

        cos_col = np.cos(data[:,col])
        cos_col = cos_col[:, np.newaxis]
        
        data = np.concatenate((data,sin_col,cos_col),axis=1)
    return data


def CT_generation(data,col_nbr):
    """Cross terms generation, add extra features to data: (feature1 x feature2)
    Args:
        data: data, shape=(N, D)
        col_nbr: list of columns
    Returns:
        data: augmented data
    """
    for index,col1 in enumerate(col_nbr):
        for col2 in col_nbr[index:]:
            new_col = np.multiply(data[:,col1],data[:,col2])
            new_col = new_col[:,np.newaxis]
            data = np.concatenate((data,new_col),axis=1)
    return data


def poly_generation(data,col_nbr,degree):
    """Add extra degree polynom to data: col^2,col^3,...col^degree
    Args:
        data: data, shape=(N, D)
        col_nbr: list of feature columns to expand, length L
    Returns:
        data: augmented data, shape=(N, D+L*(degree-1))
    """
    for col in col_nbr:
        for i in range(2,degree+1):
            new_col = np.power(data[:,col].astype(np.float32) ,i)
            new_col = new_col[:,np.newaxis]
            data = np.concatenate((data,new_col), axis=1)
    return data


def feature_generation(data, degree):
    """Add extra features to data
    Args:
        data: data, shape=(N, D)
        col_nbr: list of columns
        degree: degree of augmented polynom 
    Returns:
        expended_data: augmented data
    """
    feature_start = 2
    expanded_data = np.copy(data)
    col_nbr = np.arange(feature_start,data.shape[1])

    # Generate polynoms 
    expanded_data = poly_generation(expanded_data,col_nbr,degree)
    # Generate cross-terms
    col_nbr = np.arange(feature_start,expanded_data.shape[1])
    expanded_data = CT_generation(expanded_data,col_nbr)
    # Generate sine and cosine    
    expanded_data = sin_cos_generation(expanded_data,col_nbr)

    return expanded_data
