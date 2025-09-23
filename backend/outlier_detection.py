import numpy as np

def outlier_detection(V_copy,rejection_level,sigma_not_hat_square,cofact_res):
    '''
    This Function returns the index of the observation whose Standardised Residual is greater than the rejection level.
    Args:
        V_copy: Copy of the Residual Vector.
        rejection_level: Float value corresponding to certain significance level(Alpha) and probablity(Beta)
        sigma_not_hat_square: Aposteriori Reference Variance
        cofact_res: Cofactor Matrix of Residuals
    Return:
        idx_to_remove: List of indices of the observations whose Standardised Residual is greater than the rejection level.
    '''
    quality_of_observations = []
    print(np.sqrt(sigma_not_hat_square)*rejection_level)
    for i in range(len(V_copy)):
        quality_of_observations.append(abs(V_copy[i])/np.sqrt((cofact_res[i][i])))
    print("Quality ",quality_of_observations)
    idx_to_remove  = []
    quality = 0
    for i in range(len(quality_of_observations)):
        if quality_of_observations[i] > np.sqrt(sigma_not_hat_square)*rejection_level:
            if quality_of_observations[i]>quality:
                quality = quality_of_observations[i]
                idx_to_remove = [i]
    return idx_to_remove



    # rejection_level=3.29,
    # outlier_detection=False,
# while outlier_detection and iteration + 1 != max_iterations:
#             V_copy = V.copy().flatten()
#             cofact_res = np.linalg.inv(P) - A @ np.linalg.inv(N) @ A.T
#             idx_to_remove = outlier_detection(
#                 V_copy, rejection_level, sigma_not_hat_squared, cofact_res
#             )
            
#             L_observed = np.delete(L_observed, idx_to_remove, axis=0)
#             L_not = np.delete(L_not, idx_to_remove, axis=0)
#             L = np.delete(L, idx_to_remove, axis=0)
#             for index in sorted(idx_to_remove, reverse=True):
#                 observations_eq.pop(index)
#             # Compute the new weight matrix
#             # Remove row i
#             P= np.delete(P, idx_to_remove, axis=0)
#             # Remove column i
#             P = np.delete(P, idx_to_remove, axis=1)
#             if len(idx_to_remove) != 0: 
#                 break
