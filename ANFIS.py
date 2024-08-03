import simpful as fuzzy
import itertools 
import random as rd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

class ANFIS(fuzzy.FuzzySystem):
    def __init__(self, 
                 variables, 
                 operators=None, 
                 show_banner=False, 
                 sanitize_input=False, 
                 verbose=True, 
                 random_state=None,
                 validation_size=0.1,
                 num_epochs=200,
                 learning_rate=0.05,):
        '''Initializes the ANFIS system. Defines important attributes of the class using the dictionary variables.
        
        Args:
          variables: dictionary with keys 'inputs' and 'outputs'. Each of these keys are also dictionaries, with keys 
                     representing the names of the inputs and outputs, respectively. Each input and output should have 
                     a dictionary as its value, containing 'n_sets', 'terms' and 'universe_of_discourse' of each variable.
        '''
        
        super().__init__(operators, show_banner, sanitize_input, verbose)
        
        self.variables = variables
        self.num_input_variables = len(variables['inputs'])
        self.name_input_variables = list(variables['inputs'].keys())
        self.info_input_variables = list(variables['inputs'].values())
        self.num_output_variables = len(variables['outputs'])
        self.name_output_variables = list(variables['outputs'].keys())
        self.info_output_variables = list(variables['outputs'].values())
        
        self._random_state = random_state
        self._validation_size = validation_size
        self._num_epochs = num_epochs
        self._learning_rate = learning_rate
        
        num_rules = self.num_output_variables
        
        for variable in self.info_input_variables:
            num_rules *= variable['n_sets']
        
        self.num_rules = num_rules
        
    def _create_variables(self):
        '''Automatically creates linguistic variables for all inputs and outputs of the system. By default, it creates
        triangular fuzzy sets that are normalized and symmetrical over the universe of discourse of the variable. All
        parameters are obtained via self.variables dictionary. Creates self.input_variables and self.output_variables,
        which contain simpful.LinguisticVariable objects, that can be plotted. To name the x-axis, simply modify the 
        atribute ._concept of the variable when using its method .plot.
        '''
        self.input_variables = []
        self.output_variables = []
        
        for i in range(self.num_input_variables):
            self.input_variables.append(fuzzy.AutoTriangle(
                n_sets=self.info_input_variables[i]['n_sets'], 
                terms=self.info_input_variables[i]['terms'], 
                universe_of_discourse=self.info_input_variables[i]['universe_of_discourse']))
            
            self.add_linguistic_variable(self.name_input_variables[i], fuzzy.AutoTriangle(
                n_sets=self.info_input_variables[i]['n_sets'], 
                terms=self.info_input_variables[i]['terms'], 
                universe_of_discourse=self.info_input_variables[i]['universe_of_discourse']))
            
        for i in range(self.num_output_variables):
            self.output_variables.append(fuzzy.AutoTriangle(
                n_sets=self.info_output_variables[i]['n_sets'], 
                terms=self.info_output_variables[i]['terms'], 
                universe_of_discourse=self.info_output_variables[i]['universe_of_discourse']))
            
            self.add_linguistic_variable(self.name_output_variables[i], fuzzy.AutoTriangle(
                n_sets=self.info_output_variables[i]['n_sets'], 
                terms=self.info_output_variables[i]['terms'], 
                universe_of_discourse=self.info_output_variables[i]['universe_of_discourse']))
            
    def _create_rules(self):
        '''Creates all possible rules using the input and output variables. All rules are of the form:
        'IF (input1 IS set1) AND ... AND (inputN IS setN) THEN (output IS y)'. In total, we have a total of 
        |input1|*...*|inputN| combinations of inputs and its fuzzy sets, where |input| is the number of fuzzy
        sets the input has, and with M outputs, we have a total of |input1|*...*|inputN|*M rules.
        '''
        
        variable_association = {}

        for i in range(self.num_input_variables):
            variable = self.name_input_variables[i]
            terms = self.info_input_variables[i]['terms']
            cartesian_product = itertools.product([variable], terms)
            variable_association[variable] = list(cartesian_product)

        for pairs in list(variable_association.values()):
            for i in range(len(pairs)):
                pairs[i] = f'({pairs[i][0]} IS {pairs[i][1]})'

        antecedents = list(itertools.product(*[pairs for pairs in list(variable_association.values())]))

        for i in range(int(self.num_rules / self.num_output_variables)):
            antecedent = f'IF ' 
            
            for j in range(len(antecedents[i]) - 1):
                antecedent += antecedents[i][j] + ' AND '
                
            antecedents[i] = antecedent + antecedents[i][-1]
            
        consequents = []

        for i in range(self.num_rules):
            consequent = f' THEN ({self.name_output_variables[i % self.num_output_variables]} IS y{i})'
            consequents.append(consequent)
            
        rules = []

        for i in range(self.num_rules):
            antecedent = antecedents[i // self.num_output_variables]
            consequent = consequents[i]
            rules.append(antecedent + consequent)
            
        coefficients = [f'a{i}' for i in range(self.num_rules * (self.num_input_variables + 1))]
        output_functions = {}

        for i in range(self.num_rules):
            output_functions[f'y{i}'] = ''
            
            for j in range(self.num_input_variables):
                output_functions[f'y{i}'] += coefficients[i : : self.num_rules][j] + ' * ' + self.name_input_variables[j] + ' + '
                
            output_functions[f'y{i}'] += coefficients[i : : self.num_rules][-1]

        for i in range(self.num_rules):
            self.set_output_function(f'y{i}', output_functions[f'y{i}'])
            self.add_rules([rules[i]])
            
        self.rules = rules
        self.coefficients = coefficients
        self.output_functions = output_functions
        
    def _predict(self, x, coefficients): 
        '''Predict method used for optimization in fitting the data. 
        
        Args:
          x: numpy array with length (num_input_variables)
          coefficients: numpy array containing numerical values for the coefficients
          
        Returns:
          y_pred: prediction generated by the model
        '''
                
        for i in range(len(self.coefficients)):
            self.set_constant(f"a{i}", coefficients[i])

        for i in range(len(self.name_input_variables)):
            self.set_variable(self.name_input_variables[i], x[i])
        
        return y_pred   
    
    ##################################################################################

    def _gradient(self, parameter, y, y_pred, firing_strength, xi=None):
        if parameter == 'a':
            gradient = wi * (y_pred - y)
        elif parameter == 'b':
            gradient = xi * wi * (y - y_pred)
        return gradient
            
    
    def fit(self, X, y):
        '''Stores input data, X, and output data, Y. (ALL OPTIMIZATION HERE)
        
        Args:
          X: numpy array with shape (num_input_data, num_input_variables)
          y: numpy array with shape (num_output_data, num_output_variables)
        '''
        self._create_variables()
        self._create_rules()
        
        x_train, x_test, y_train, y_val = train_test_split(
                    X, y, test_size=self._validation_size, random_state=self._random_state)
        
        coefficient_values = np.array([rd.random() for _ in range(len(self.coefficients))])
        train_loss = []
        validation_loss = []
        
        for _ in range(self._num_epochs): 
            t_loss, v_loss = 0, 0
            
            for i in range(x_train.shape[0]):
                y_pred = self._predict(x_train[i], coefficient_values)
                t_loss += 0.5 * (y_pred - y_train[i])**2
                
                gradient = np.zeros(len(coefficient_values))
                
                for j in range(len(coefficient_values)):
                    if (j % (self.num_input_variables + 1) ) == self.num_input_variables:
                        gradient[j] += self._gradient('b', y_train[j], y_pred, 
                                                      self.get_firing_strengths()[self.num_rules // j])
                        
                    else:
                        gradient[j] += self._gradient('a', y_train[j], y_pred, 
                                                      self.get_firing_strengths()[self.num_rules // j], 
                                                      xi=x_train[i][AAAAAAAAAAAAA])
                        
                        
                coefficient_values += - gradient * learning_rate
                
            for i in range(x_val.shape[0]):
                y_pred = self._predict(x_val[i], coefficient_values)
                v_loss += 0.5 * (y_pred - y_val[i])**2
                
            train_loss.append(t_loss)  
            validation_loss.append(v_loss)
          
        self.coefficient_values = coefficient_values
            
    
                    
                    
            #calculate backpropagation with respect to train loss
            #atualization of coefficent_values -> coefficient_values = [gradient]
            #criteria for early stopping
            #criteria for removing rules
        
            
        
        

    
            
    def predict(self, X):         
        for i in range(len(self.coefficients)):
            self.set_constant(f"a{i}", self.coefficient_values[i])

        for i in range(len(self.name_input_variables)):
            self.set_variable(self.name_input_variables[i], X[i])

        y_pred = np.array(list(self.Sugeno_inference().values()))

        return y_pred