import sys
import os
import re
import unicodedata
from codecs import open
from math import isnan, isinf
from abc import ABCMeta, abstractmethod

from fuzzywuzzy import fuzz
#from stanfordnlp.server import CoreNLPClient


import matplotlib.pyplot as plt


import math
import sqlite3

def execute_example2(example, canon_string=False):
    sql = example['sql2']
    db_file = f"/media/sunveil/Data/header_detection/poddubnyy/postgraduate/squall/tables/db/{example['tbl']}.db"
    connection = sqlite3.connect(db_file)
    c = connection.cursor()
    c.execute(sql)
    answer_list = list()
    for result, in c:
        result = str(result)
        answer_list.append(result)

    if not canon_string:
        execution_value = to_value_list(answer_list)
        
    if canon_string:
        ex_id = example['nt']
        canon_strings = canon_strings_map[ex_id]
        execution_value = to_value_list(answer_list, canon_strings)
        
    return execution_value

def fuzzy_matching(str1, str2):
    return fuzz.ratio(str1, str2)

def strict_denotation_accuracy(target_values, predicted_values):

    """Return True if the predicted denotation is correct.

    Args:
        target_values (list[Value])
        predicted_values (list[Value])
    Returns:
        bool
    """
    # Check size
    if len(target_values) != len(predicted_values):
        return False
    # Check items
    for target in target_values:
        if not any(target.match(pred) for pred in predicted_values):
            return False
    return True



def plot_and_save_model_performance(results, file_path):
    # Parsing the checkpoint numbers and metrics from the results
    checkpoints = []
    fuzzy_match = []
    strict_denotation_accuracy_exec = []
    flexible_denotation_accuracy_exec = []

    for key, value in results.items():
        if not key.startswith('test'):
            checkpoint_str = key.split('-')[-1]
            checkpoint = int(checkpoint_str)
            checkpoints.append(checkpoint)
            fuzzy_match.append(value['Fuzzy_Match'])
            strict_denotation_accuracy_exec.append(value['Strict_Denotation_Accuracy_Exec'])
            flexible_denotation_accuracy_exec.append(value['Flexible_Denotation_Accuracy_Exec'])

    # Sorting data based on checkpoint numbers
    sorted_indices = sorted(range(len(checkpoints)), key=lambda k: checkpoints[k])
    checkpoints = [checkpoints[i] for i in sorted_indices]
    fuzzy_match = [fuzzy_match[i] for i in sorted_indices]
    strict_denotation_accuracy_exec = [strict_denotation_accuracy_exec[i] for i in sorted_indices]
    flexible_denotation_accuracy_exec = [flexible_denotation_accuracy_exec[i] for i in sorted_indices]

    # Creating the plot
    fig, ax1 = plt.subplots()

    # Plotting Fuzzy Match on the primary axis
    ax1.set_xlabel('Checkpoint')
    ax1.set_ylabel('Fuzzy Match', color='tab:blue')
    ax1.plot(checkpoints, fuzzy_match, label='Fuzzy Match', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Creating a second y-axis for Strict and Flexible Denotation Accuracy Exec
    ax2 = ax1.twinx()
    ax2.set_ylabel('Strict & Flexible Denotation Accuracy Exec', color='tab:red')
    ax2.plot(checkpoints, strict_denotation_accuracy_exec, label='Strict Denotation Accuracy Exec', color='tab:red')
    ax2.plot(checkpoints, flexible_denotation_accuracy_exec, label='Flexible Denotation Accuracy Exec', color='tab:pink')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Adding a legend
    fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))

    # Show plot
    plt.title('Model Performance Metrics over Checkpoints')

    # Saving the figure
    fig.savefig(file_path)



def flexible_denotation_accuracy(target_values, predicted_values):
    
    units = ['year',"episodes","m","million","weeks","week","mm","nd","th","days",
         "years", "events", "£", "miles", "kg", "$", "days",",",
        "losses","l","season"] 
    

    # Create a set of valid target values
    def remove_units(input_str):
        for unit in units:
            input_str = input_str.replace(unit, "").strip().lower()
            
        try:
            return float(input_str)
        except ValueError:
            return input_str
        
    try : 
        normalized_target_values = [remove_units(tv) for tv in target_values]
        normalized_predicted_values = [remove_units(pv) for pv in predicted_values]
        if normalized_target_values==normalized_predicted_values:
            return True
    except:
        pass
    target_values = to_value_list(target_values)
    predicted_values = to_value_list(predicted_values)
    
    
    valid_targets = [to_value_list([c]) for c in ["none","no","older","lower","less","below","false","f"]]
    if predicted_values in valid_targets and target_values in valid_targets:
        return True
    
    valid_targets = [to_value_list([c]) for c in ["1","more","yes","above","true","higher","first","once"]]
    if predicted_values in valid_targets and target_values in valid_targets:
        return True
    
    if len(target_values) != len(predicted_values):
        return False
    
    else:
        for target in target_values:
            if not any(target.match(pred) for pred in predicted_values):
                return False
            
    return True


################ String Normalization ################

def normalize(x):
    if not isinstance(x, str):
        x = x.decode('utf8', errors='ignore')
    # Remove diacritics
    x = ''.join(c for c in unicodedata.normalize('NFKD', x)
                if unicodedata.category(c) != 'Mn')
    # Normalize quotes and dashes
    x = re.sub(r"[‘’´`]", "'", x)
    x = re.sub(r"[“”]", "\"", x)
    x = re.sub(r"[‐‑‒–—−]", "-", x)
    while True:
        old_x = x
        # Remove citations
        x = re.sub(r"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())
        # Remove details in parenthesis
        x = re.sub(r"(?<!^)( \([^)]*\))*$", "", x.strip())
        # Remove outermost quotation mark
        x = re.sub(r'^"([^"]*)"$', r'\1', x.strip())
        if x == old_x:
            break
    # Remove final '.'
    if x and x[-1] == '.':
        x = x[:-1]
    # Collapse whitespaces and convert to lower case
    x = re.sub(r'\s+', ' ', x, flags=re.U).lower().strip()
    return x


################ Value Types ################

class Value(object):
    __metaclass__ = ABCMeta

    # Should be populated with the normalized string
    _normalized = None

    @abstractmethod
    def match(self, other):
        """Return True if the value matches the other value.
        Args:
            other (Value)
        Returns:
            a boolean
        """
        pass

    @property
    def normalized(self):
        return self._normalized


class StringValue(Value):

    def __init__(self, content):
        assert isinstance(content, str)
        self._normalized = normalize(content)
        self._hash = hash(self._normalized)

    def __eq__(self, other):
        return isinstance(other, StringValue) and self.normalized == other.normalized

    def __hash__(self):
        return self._hash

    def __str__(self):
        return 'S' +  str([self.normalized])
    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        return self.normalized == other.normalized


class NumberValue(Value):

    def __init__(self, amount, original_string=None):
        assert isinstance(amount, (int, float))
        if abs(amount - round(amount)) < 1e-6:
            self._amount = int(amount)
        else:
            self._amount = float(amount)
        if not original_string:
            self._normalized = unicode(self._amount)
        else:
            self._normalized = normalize(original_string)
        self._hash = hash(self._amount)

    @property
    def amount(self):
        return self._amount

    def __eq__(self, other):
        return isinstance(other, NumberValue) and self.amount == other.amount

    def __hash__(self):
        return self._hash

    def __str__(self):
        return ('N(%f)' % self.amount) + str([self.normalized])
    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, NumberValue):
            return abs(self.amount - other.amount) < 1e-6
        return False

    @staticmethod
    def parse(text):
        """Try to parse into a number.
        Return:
            the number (int or float) if successful; otherwise None.
        """
        try:
            return int(text)
        except:
            try:
                amount = float(text)
                assert not isnan(amount) and not isinf(amount)
                return amount
            except:
                return None


class DateValue(Value):

    def __init__(self, year, month, day, original_string=None):
        """Create a new DateValue. Placeholders are marked as -1."""
        assert isinstance(year, int)
        assert isinstance(month, int) and (month == -1 or 1 <= month <= 12)
        assert isinstance(day, int) and (day == -1 or 1 <= day <= 31)
        assert not (year == month == day == -1)
        self._year = year
        self._month = month
        self._day = day
        if not original_string:
            self._normalized = '{}-{}-{}'.format(
                year if year != -1 else 'xx',
                month if month != -1 else 'xx',
                day if day != '-1' else 'xx')
        else:
            self._normalized = normalize(original_string)
        self._hash = hash((self._year, self._month, self._day))

    @property
    def ymd(self):
        return (self._year, self._month, self._day)

    def __eq__(self, other):
        return isinstance(other, DateValue) and self.ymd == other.ymd

    def __hash__(self):
        return self._hash

    def __str__(self):
        return (('D(%d,%d,%d)' % (self._year, self._month, self._day))
                + str([self._normalized]))
    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, DateValue):
            return self.ymd == other.ymd
        return False

    @staticmethod
    def parse(text):
        """Try to parse into a date.
        Return:
            tuple (year, month, date) if successful; otherwise None.
        """
        try:
            ymd = text.lower().split('-')
            assert len(ymd) == 3
            year = -1 if ymd[0] in ('xx', 'xxxx') else int(ymd[0])
            month = -1 if ymd[1] == 'xx' else int(ymd[1])
            day = -1 if ymd[2] == 'xx' else int(ymd[2])
            assert not (year == month == day == -1)
            assert month == -1 or 1 <= month <= 12
            assert day == -1 or 1 <= day <= 31
            return (year, month, day)
        except:
            return None


################ Value Instantiation ################

def to_value(original_string, corenlp_value=None):
    """Convert the string to Value object.
    Args:
        original_string (basestring): Original string
        corenlp_value (basestring): Optional value returned from CoreNLP
    Returns:
        Value
    """
    if isinstance(original_string, Value):
        # Already a Value
        return original_string
    if not corenlp_value:
        corenlp_value = original_string
    # Number?
    amount = NumberValue.parse(corenlp_value)
    if amount is not None:
        return NumberValue(amount, original_string)
    # Date?
    ymd = DateValue.parse(corenlp_value)
    if ymd is not None:
        if ymd[1] == ymd[2] == -1:
            return NumberValue(ymd[0], original_string)
        else:
            return DateValue(ymd[0], ymd[1], ymd[2], original_string)
    # String.
    return StringValue(original_string)

def to_value_list(original_strings, corenlp_values=None):
    """Convert a list of strings to a list of Values
    Args:
        original_strings (list[basestring])
        corenlp_values (list[basestring or None])
    Returns:
        list[Value]
    """
    assert isinstance(original_strings, (list, tuple, set))
    if corenlp_values is not None:
        assert isinstance(corenlp_values, (list, tuple, set))
        #assert len(original_strings) == len(corenlp_values)
        return list(set(to_value(x, y) for (x, y)
                in zip(original_strings, corenlp_values)))
    else:
        return list(set(to_value(x) for x in original_strings))


################ Check the Predicted Denotations ################




################ Batch Mode ################

def tsv_unescape(x):
    """Unescape strings in the TSV file.
    Escaped characters include:
        newline (0x10) -> backslash + n
        vertical bar (0x7C) -> backslash + p
        backslash (0x5C) -> backslash + backslash
    Args:
        x (str or unicode)
    Returns:
        a unicode
    """
    return x.replace(r'\n', '\n').replace(r'\p', '|').replace('\\\\', '\\')

def tsv_unescape_list(x):
    """Unescape a list in the TSV file.
    List items are joined with vertical bars (0x5C)
    Args:
        x (str or unicode)
    Returns:
        a list of unicodes
    """
    return [tsv_unescape(y) for y in x.split('|')]

tagged_dataset_path = "/media/sunveil/Data/header_detection/poddubnyy/postgraduate/squall/tables/tagged/"
database_path = "/media/sunveil/Data/header_detection/poddubnyy/postgraduate/squall/tables/db/"
#corenlp_path = "../data/stanford-corenlp-full-2018-10-05/"
            
#os.environ['CORENLP_HOME'] = corenlp_path
db_path = database_path
target_values_map = {} 
canon_strings_map = {}
for filename in os.listdir(tagged_dataset_path):
    filename = os.path.join(tagged_dataset_path, filename)
    #print(sys.stderr, 'Reading dataset from', filename)
    with open(filename, 'r', 'utf8') as fin:
        header = fin.readline().rstrip('\n').split('\t')
        for line in fin:
            stuff = dict(zip(header, line.rstrip('\n').split('\t')))
            ex_id = stuff['id']
            original_strings = tsv_unescape_list(stuff['targetValue'])
            canon_strings = tsv_unescape_list(stuff['targetCanon'])
            target_values_map[ex_id] = to_value_list(
                    original_strings, canon_strings)
            canon_strings_map[ex_id] = canon_strings
            
