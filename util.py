def adjust_probs(prob, porportion_climo, porportion_train):
    adjusted_prob=(prob*(porportion_climo/porportion_train))/((prob*(porportion_climo/porportion_train))+((1-prob)*((1-porportion_climo)/(1-porportion_train))))
    return adjusted_prob
