import numpy as np

class Fuzzy_number():
    def __init__(self, closure, number_alphalevels, alpha_interval=(0, 1)):
        '''Initializes a generic fuzzy number with closure given by a list. For the operations to work, it is required
        that number_alphalevels in both objects are equal. 
        
        Args:
          closure: tupple or list representing the closure of the number.
          
          number_alphalevels: integer representing the number of alphalevels created.
          
          alpha_interval: tuple or list representing the alpha values.
          
        Atributes:
          self.alphas: numpy array of alphas between 0 and 1.
          
          self.alphalevels: numpy array containing the alpha levels of the number, which are paired with the alphas in
                            self.alphas, for a possible plot.
        '''
        self.number_alphalevels = number_alphalevels
        self.alpha_min = alpha_interval[0]
        self.alpha_max = alpha_interval[1]
        self.closure = closure
        self.alphas = np.linspace(self.alpha_min, self.alpha_max, self.number_alphalevels)
        self.alphalevels = []
        
    def triangular(self, u):
        '''Creates a triangular fuzzy number around the value u.

        Args:
          u: value representing the peak of the fuzzy number
        '''
        if u > self.closure[1] or u < self.closure[0]:
            return 'ERROR: The peak of the fuzzy number must be contained in its closure.'
        
        else:
            lower_limit, upper_limit = self.closure[0], self.closure[1] 
            self.alphalevels = np.array([[(u - lower_limit) * alpha + lower_limit, 
                                          (u - upper_limit) * alpha + upper_limit] for alpha in self.alphas])
    
    def trapezoidal(self, a, b):
        '''Creates a trapezoidal fuzzy number. The arguments represent the values that define the trapezoid, i.e.,
        closure[0] to a is a positive linear function, a to b is constant at 1, and b to closure[1] is a negative linear
        function.

        Args:
          a, b: parameters of the trapezoid.
        '''  
        if a < self.closure[0] or b > self.closure[1]:
            return 'ERROR: The peak of the fuzzy number must be contained in its closure.'
        
        elif a > b:
            return 'ERROR: The interval given is not in the correct order.'
        
        else:
            self.alphalevels = np.array([
                [(a - self.closure[0]) * alpha + self.closure[0], 
                 (b - self.closure[1]) * alpha + self.closure[1]] for alpha in self.alphas])

    def quadratic(self, a):
        '''Creates a quadratic fuzzy number. The characteristic function should be written in the form:
        f(x) = a(x - self.closure[0])(x - self.closure[1]).

        Args:
          a: quadratic coefficient of the characteristic function.
        '''  
        if a >= 0:
            return 'ERROR: The parabola must be concave down.'
        
        else:
            b = - a * (self.closure[0] + self.closure[1])
            c = a * self.closure[0] * self.closure[1]
            
            if - (b**2 - 4 * a * c) / (4 * a) < 1:
                return 'ERROR: The fuzzy number must have alpha levels in alpha=1.'
            else:
                self.alphalevels = np.array([
                    [(- b - np.sqrt(b**2 - 4 * a * (c - alpha))) / (2 * a), 
                     (- b + np.sqrt(b**2 - 4 * a * (c - alpha))) / (2 * a)] for alpha in self.alphas])

    def gaussian(self, a):
        '''Creates a gaussian fuzzy number around u, the middle point of the closure. The equation has the form: 
        exp(- (x-u)**2 / a).

        Args:
          a: parameter of the gaussian equation.
        '''  
        u = (self.closure[1] - self.closure[0]) / 2
        delta = abs(self.closure[1] - u)
        alpha_levels = []
        
        for alpha in self.alphas:
            if alpha >= np.exp(- (delta / a)**2):
                alpha_levels.append([u - np.sqrt(np.log(1 / alpha ** (a**2))), 
                                     u + np.sqrt(np.log(1 / alpha ** (a**2)))])
                
            else:
                alpha_levels.append([u - delta, u + delta])
  
        self.alphalevels = np.array(alpha_levels)
    
    def __add__(self, fuzzy_number):
        '''Addition of two fuzzy numbers.
        
        Args:
          fuzzy_number: instance of class Fuzzy_number.
          
        Returns:
          result: instance of class Fuzzy_number.
        ''' 
        closure = (self.alphalevels[0][0] + fuzzy_number.alphalevels[0][0], 
                   self.alphalevels[0][1] + fuzzy_number.alphalevels[0][1])
        
        result = Fuzzy_number(closure, self.number_alphalevels)
        result.alphalevels = self.alphalevels + fuzzy_number.alphalevels
        return result
    
    def __sub__(self, fuzzy_number):
        '''Standard subtraction of two fuzzy numbers.
        
        Args:
          fuzzy_number: instance of class Fuzzy_number.
          
        Returns:
          result: instance of class Fuzzy_number.
        '''    
        closure = (self.alphalevels[0][0] - fuzzy_number.alphalevels[0][1], 
                   self.alphalevels[0][1] - fuzzy_number.alphalevels[0][0])
        
        result = Fuzzy_number(closure, self.number_alphalevels)
        
        transpose = self.alphalevels.T
        lower_limits, upper_limits = transpose[0], transpose[1]
        result.alphalevels = self.alphalevels - np.array([upper_limits, lower_limits]).T
        
        return result
    
    def __mul__(self, c):
        '''Multiplication by a constant.
        
        Args:
          c: float representing a constant.
          
        Returns:
          result: instance of class Fuzzy_number.
        '''
        if c >= 0:
            alphalevels = c * self.alphalevels

            closure = (c * self.alphalevels[0][0], c * self.alphalevels[0][1])
            
        else:
            transpose = self.alphalevels.T
            lower_limits, upper_limits = transpose[0], transpose[1]
            alphalevels = c * np.array([upper_limits, lower_limits]).T

            closure = (c * self.alphalevels[0][1], c * self.alphalevels[0][0])
        
        result = Fuzzy_number(closure, self.number_alphalevels)
        result.alphalevels = alphalevels
        return result
    
    def __rmul__(self, c):
        '''Defines simmetry in multiplication.
        
        Args:
          c: float representing a constant.
          
        Returns:
          result: instance of class Fuzzy_number.
        '''
        return self.__mul__(c)
    
    def __matmul__(self, fuzzy_number):
        '''Multiplication of two fuzzy numbers.
        
        Args:
          fuzzy_number: instance of class Fuzzy_number.
          
        Returns:
          result: instance of class Fuzzy_number.
        '''    
        alphalevels = []
        
        for i in range(self.number_alphalevels):
            limits = np.outer(self.alphalevels[i], fuzzy_number.alphalevels[i]).ravel()
            lower_limit = np.min(limits)
            upper_limit = np.max(limits)
            alphalevel = [lower_limit, upper_limit]
            alphalevels.append(alphalevel)
            
        closure = (alphalevels[0][0], alphalevels[0][1])
        result = Fuzzy_number(closure, self.number_alphalevels)
        result.alphalevels = np.array(alphalevels)
        return result
    
    def __truediv__(self, fuzzy_number):
        '''Division of two fuzzy numbers.
        
        Args:
          fuzzy_number: instance of class Fuzzy_number.
          
        Returns:
          result: instance of class Fuzzy_number.
        '''  
        if 0 in fuzzy_number.alphalevels:
            return 'ERROR: This operation is not possible, because 0 is in the denominator.'
        
        else: 
            transpose = fuzzy_number.alphalevels.T
            lower_limits, upper_limits = transpose[0], transpose[1]
            denominator_alphalevels = (1 / np.array([upper_limits, lower_limits]).T)
            
            closure_limits = np.outer(self.alphalevels[0], denominator_alphalevels[0]).ravel()
            lower_limit = np.min(closure_limits)
            upper_limit = np.max(closure_limits)
            closure = (lower_limit, upper_limit)
            
            denominator = Fuzzy_number(closure, self.number_alphalevels)
            denominator.alphalevels = denominator_alphalevels
            
            result = self.__matmul__(denominator)
        return result
            
            
            