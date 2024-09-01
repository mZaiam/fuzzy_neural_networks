import simpful as fuzzy
import itertools 
import random as rd
import numpy as np

from sklearn.model_selection import train_test_split

class ANFIS(fuzzy.FuzzySystem):
    def __init__(self, 
                 variables, 
                 operators=['AND_PRODUCT'], 
                 rules=None,
                 fuzzy_sets=None,
                 show_banner=False, 
                 sanitize_input=False, 
                 verbose=True, 
                 random_state=None,
                 learning_algorithm='OnlineLearning',
                 #num_batches=5,
                 num_epochs=20,
                 learning_rate=0.01,
                 beta=0,
                 ):
        
        '''Initializes the ANFIS system. Defines important attributes of the class using the dictionary variables.
        
        Args:
          variables: dictionary with keys 'inputs' and 'outputs'. Each of these keys are also dictionaries, with keys 
                     representing the names of the inputs and outputs, respectively. Each input and output should have 
                     a dictionary as its value, containing 'n_sets', 'terms' and 'universe_of_discourse' of each variable.
          rules: list of rules of the system, to be inserted manually.
          fuzzy_sets: dictionary with the keys being the variables names in strings, and the values being
                      a list of fuzzy sets.
        '''
        
        super().__init__(operators, show_banner, sanitize_input, verbose)
        
        self._variables = variables
        self._num_input_variables = len(variables['inputs'])
        self._name_input_variables = list(variables['inputs'].keys())
        self._info_input_variables = list(variables['inputs'].values())
        self._num_output_variables = len(variables['outputs'])
        self._name_output_variables = list(variables['outputs'].keys())
        self._info_output_variables = list(variables['outputs'].values())
        
        self.fuzzy_sets = fuzzy_sets
        self.rules = rules
        
        self._random_state = random_state
        self._learning_algorithm = learning_algorithm
        self._num_epochs = num_epochs
        self._learning_rate = learning_rate
        self._beta = beta
        
        if not isinstance(rules, list):
            num_rules = self._num_output_variables

            for variable in self._info_input_variables:
                num_rules *= variable['n_sets']

            self._num_rules = num_rules
            
        else: 
            self.rules = rules
            self._num_rules = len(rules)
        
    def _create_variables(self):
        '''Automatically creates linguistic variables for all inputs and outputs of the system. By default, it creates
        triangular fuzzy sets that are normalized and symmetrical over the universe of discourse of the variable. All
        parameters are obtained via self._variables dictionary. Creates self.input_variables and self.output_variables,
        which contain simpful.LinguisticVariable objects, that can be plotted. To name the x-axis, simply modify the 
        atribute ._concept of the variable when using its method .plot.
        '''
        
        self.input_variables = []
        self.output_variables = []
        
        if not isinstance(self.fuzzy_sets, dict):
            for i in range(self._num_input_variables):
                self.input_variables.append(fuzzy.AutoTriangle(
                    n_sets=self._info_input_variables[i]['n_sets'], 
                    terms=self._info_input_variables[i]['terms'], 
                    universe_of_discourse=self._info_input_variables[i]['universe_of_discourse']))

                self.add_linguistic_variable(self._name_input_variables[i], fuzzy.AutoTriangle(
                    n_sets=self._info_input_variables[i]['n_sets'], 
                    terms=self._info_input_variables[i]['terms'], 
                    universe_of_discourse=self._info_input_variables[i]['universe_of_discourse']))

            for i in range(self._num_output_variables):
                self.output_variables.append(fuzzy.AutoTriangle(
                    n_sets=self._info_output_variables[i]['n_sets'], 
                    terms=self._info_output_variables[i]['terms'], 
                    universe_of_discourse=self._info_output_variables[i]['universe_of_discourse']))

                self.add_linguistic_variable(self._name_output_variables[i], fuzzy.AutoTriangle(
                    n_sets=self._info_output_variables[i]['n_sets'], 
                    terms=self._info_output_variables[i]['terms'], 
                    universe_of_discourse=self._info_output_variables[i]['universe_of_discourse']))
                
        else:
            for input_variable in self._name_input_variables:
                self.input_variables.append(
                    fuzzy.LinguisticVariable(
                        self.fuzzy_sets[input_variable], 
                        concept=input_variable, 
                        universe_of_discourse=self._variables['inputs'][input_variable]['universe_of_discourse']))
                
                self.add_linguistic_variable(
                    input_variable, 
                    fuzzy.LinguisticVariable(
                        self.fuzzy_sets[input_variable], 
                        concept=input_variable, 
                        universe_of_discourse=self._variables['inputs'][input_variable]['universe_of_discourse']))
                
            for output_variable in self._name_output_variables:
                self.input_variables.append(
                    fuzzy.LinguisticVariable(
                        self.fuzzy_sets[output_variable], 
                        concept=output_variable, 
                        universe_of_discourse=self._variables['outputs'][output_variable]['universe_of_discourse']))
                
                self.add_linguistic_variable(
                    output_variable, 
                    fuzzy.LinguisticVariable(
                        self.fuzzy_sets[output_variable], 
                        concept=output_variable, 
                        universe_of_discourse=self._variables['outputs'][output_variable]['universe_of_discourse']))

    def _create_rules(self):
        '''Creates all possible rules using the input and output variables. All rules are of the form:
        'IF (input1 IS set1) AND ... AND (inputN IS setN) THEN (output IS y)'. In total, we have a total of 
        |input1|*...*|inputN| combinations of inputs and its fuzzy sets, where |input| is the number of fuzzy
        sets the input has, and with M outputs, we have a total of |input1|*...*|inputN|*M rules. Has an option 
        manually insert the rules.
        '''
        
        if not isinstance(self.rules, list):
            variable_association = {}

            for i in range(self._num_input_variables):
                variable = self._name_input_variables[i]
                terms = self._info_input_variables[i]['terms']
                cartesian_product = itertools.product([variable], terms)
                variable_association[variable] = list(cartesian_product)

            for pairs in list(variable_association.values()):
                for i in range(len(pairs)):
                    pairs[i] = f'({pairs[i][0]} IS {pairs[i][1]})'

            antecedents = list(itertools.product(*[pairs for pairs in list(variable_association.values())]))

            for i in range(int(self._num_rules / self._num_output_variables)):
                antecedent = f'IF ' 

                for j in range(len(antecedents[i]) - 1):
                    antecedent += antecedents[i][j] + ' AND '

                antecedents[i] = antecedent + antecedents[i][-1]

            consequents = []

            for i in range(self._num_rules):
                consequent = f' THEN ({self._name_output_variables[i % self._num_output_variables]} IS y{i})'
                consequents.append(consequent)

            rules = []

            for i in range(self._num_rules):
                antecedent = antecedents[i // self._num_output_variables]
                consequent = consequents[i]
                rules.append(antecedent + consequent)
            
            self.rules = rules
          
        self._num_rules = len(self.rules)
        coefficients = np.array([f'a{i}' for i in range(self._num_rules * (self._num_input_variables + 1))])
        coefficients = coefficients.reshape((self._num_rules, -1))
        output_functions = {}

        for i in range(self._num_rules):
            output_functions[f'y{i}'] = ''
            
            for j in range(self._num_input_variables):
                output_functions[f'y{i}'] += coefficients[i][j] + ' * ' + self._name_input_variables[j] + ' + '
                
            output_functions[f'y{i}'] += coefficients[i][-1]

        for i in range(self._num_rules):
            self.set_output_function(f'y{i}', output_functions[f'y{i}'])
            self.add_rules([self.rules[i]])
            
        self._coefficients = coefficients
        self.output_functions = output_functions
                
    def _setup(self):
        '''Creates the variables and rules of the system. 
        '''
        
        self._create_variables()
        self._create_rules()
        
    def _remove_rules(self, index, rules, coefficient_values):
        '''Removes rules based on their activation strength.
        
        Args:
          index: list containing the index of the rules to be removed.
          rules: list of rules of the system.
          coefficient_values: numpy array with values for the coefficients, with size (num_rules * num_output_variables, 
                              num_input_variables + 1).
        '''
        
        self._rules = []
        self.rules = list(np.delete(np.array(rules), index, axis=0))
        
        self._create_rules()
        
        new_coefficient_values = np.delete(coefficient_values, index, axis=0)
        
        return new_coefficient_values
                
    def _predict(self, x, coefficient_values): 
        '''Prediction method used for online learning. 
        
        Args:
          x: numpy array with length (1, num_input_variables).
          coefficient_values: numpy array with values for the coefficients, with size (num_rules * num_output_variables, 
                              num_input_variables + 1).
        Returns:
          y_pred: numpy array with the outputs, with size (1, num_output_variables).
        '''
                
        for i in range(len(self._coefficients.ravel())):
            self.set_constant(f"a{i}", coefficient_values.ravel()[i])

        for i in range(self._num_input_variables):
            self.set_variable(self._name_input_variables[i], x[i])
        
        y_pred = np.array(list(self.Sugeno_inference().values()))
        
        return y_pred   
            
    def _gradient(self, parameter, y, y_pred, rule_strength, xi=None):
        '''Calculates the partial derivative of the parameters with respect to the loss function, given by:
        1/2 * (y_pred - y) ** 2.
        
        Args:
          parameter: string indicating what kind of parameter is it.
          y: output value from the dataset.
          y_pred: predicted output from the system.
          rule_strength: normalized value from the rule's firing strength.
          xi: input value from that specific parameter.
          
        Returns:
          gradient: value of the gradient of the function to the parameter selected.
        '''
                    
        if parameter == 'independent_coefficient':
            gradient = rule_strength * (y_pred - y) 
        elif parameter == 'linear_coefficient':
            gradient = xi * rule_strength * (y_pred - y) 
        return gradient
    
    def fit(self, x_train, x_val, y_train, y_val):
        '''Performs online learning in the system.
        
        Args:
          x_train, x_val, y_train, y_val: numpy arrays for the training of the model.
        '''
        
        self._setup()
        
        coefficient_values = np.random.uniform(low=-1, high=1, size=self._coefficients.shape)
        train_loss = []
        validation_loss = []
                
        for _ in range(self._num_epochs): 
            t_loss, v_loss = 0, 0
            rule_strength = []
            batch_gradient = np.zeros(self._coefficients.shape)

            for i in range(x_train.shape[0]):
                y_pred = self._predict(x_train[i], coefficient_values)
                t_loss += 0.5 * (y_pred - y_train[i])**2
                
                rule_strength.append(self.get_firing_strengths())
                
                gradient = np.zeros(self._coefficients.shape)
                
                for j in range(self._num_rules):
                    for k in range(self._num_input_variables + 1):
                        if k == self._num_input_variables:
                            gradient[j][k] += self._gradient(
                                'independent_coefficient', 
                                 y_train[i], 
                                 y_pred, 
                                 self.get_firing_strengths()[j] / sum(self.get_firing_strengths())
                                 )
                        else:
                            gradient[j][k] += self._gradient(
                                'linear_coefficient', 
                                 y_train[i], 
                                 y_pred, 
                                 self.get_firing_strengths()[j] / sum(self.get_firing_strengths()),
                                 xi=x_train[i][k]
                                 )
                            
                    if self._learning_algorithm == 'OnlineLearning':
                        coefficient_values += - gradient * self._learning_rate
                        
                    elif self._learning_algorithm == 'BatchLearning':
                        batch_gradient += gradient
                        
                if self._learning_algorithm == 'BatchLearning':
                    coefficient_values += - (batch_gradient / x_train.shape[0]) * self._learning_rate
                
            for i in range(x_val.shape[0]):
                y_pred = self._predict(x_val[i], coefficient_values)
                v_loss += 0.5 * (float(y_pred) - y_val[i])**2
                
            train_loss.append(t_loss / x_train.shape[0])  
            validation_loss.append(v_loss / x_val.shape[0])
            
            coefficient_values = np.clip(coefficient_values, -1, 1)
            
            mean_rule_strength = np.mean(rule_strength, axis=0)

            if np.sum(mean_rule_strength <= self._beta) != 0:
                index = [i for i in range(len(mean_rule_strength)) if mean_rule_strength[i] <= self._beta]
                coefficient_values = self._remove_rules(index, self.rules, coefficient_values)
                
        self._coefficient_values = coefficient_values
        self.train_loss = train_loss
        self.validation_loss = validation_loss       
    
    def predict(self, x): 
        '''Prediction method. 
        
        Args:
          x: numpy array with size (n, num_input_variables).
          
        Returns:
          y_pred: numpy array with the outputs, with size (n, num_output_variables).
        '''
        
        if self._coefficient_values is None:
            return 'ERROR: The ANFIS is not trained yet.'
        
        for i in range(len(self._coefficients.ravel())):
            self.set_constant(f"a{i}", self._coefficient_values.ravel()[i])

        y_pred = []
        
        for xi in x:
            for j in range(self._num_input_variables):
                self.set_variable(self._name_input_variables[j], xi[j])

            y_pred.append(list(self.Sugeno_inference().values()))

        return np.array(y_pred).ravel()