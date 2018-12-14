#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 20:30:20 2018

@author: grantrob
"""

#Advent of Code: Problem 1

#import requests
#
#def new_session(day):
#    session = requests.Session()
#    url = 'https://adventofcode.com/2018/day/{}/input'
#    session.get(url)
#    session.post()

import string
import math
import itertools
import functools
import time
from collections import Counter, defaultdict
import re
import logging
import pdb
logging.basicConfig(filename='output.log',level=logging.DEBUG)

def input_to_list(stream, function=int, delimiter='\n'):
    data = [function(i) for i in stream.split(delimiter)]
    return data

def problem_one(stream):
    data = input_to_list(stream)
    data_sum = sum(data)
    return data_sum

def problem_two(stream):
    data = input_to_list(stream)
    shifts = itertools.cycle(data)
    totals = itertools.accumulate(shifts)
    discovered = {}
    for total in totals:
        if total in discovered: 
            return total
        else:
            discovered[total] = 0
            
def problem_three(stream):
    data = input_to_list(stream, function=str)
    counts = [Counter(barcode) for barcode in data]
    doubles = sum(1 for c in counts if 2 in c.values())
    triples = sum(1 for c in counts if 3 in c.values())
    return doubles * triples

def problem_four(stream):
    def common_letters(first, second):
        length = len(first)
        for i in range(length):
            left = first[:i] + first[i+1:]
            right = second[:i] + second[i+1:]
            if left == right:
                return left
        
        return False
            
    data = input_to_list(stream, function=str)
    count_pairs = [(barcode, Counter(barcode)) for barcode in data]
    possibilities = []
    for first in count_pairs:
        for second in count_pairs:
            f_word, f_counts = first
            s_word, s_counts = second
            if f_word != s_word:
                result = f_counts - s_counts
                if len(result) == 1:
                    pair = f_word, s_word
                    possibilities.append(pair)
    
    for possibility in possibilities:
        left, right = possibility
        result = common_letters(left, right)
        if result:
            return result
        
def determine_boundaries(claim):
        _, indices, dimensions = claim
        i_start, j_start = indices
        i_, j_ = dimensions
        i_end, j_end = i_start + i_, j_start + j_
        return ((i_start, i_end), (j_start, j_end))
    
def problem_five_six(stream):
    def generate_table(length):
        table = [[[] for i in range(length)] for j in range(length)]
        return table
    
    def increment_cell(payload, table):
        number, location = payload
        i, j = location
        table[i][j].append(number)

    def convert_input(claim):
        number, _, indices, dimensions = claim.split()
        number = int(number[1:])
        indices = tuple(int(i) for i in indices[:-1].split(','))
        dimensions = tuple(int(i) for i in dimensions.split('x'))
        return number, indices, dimensions
    
    def apply_increments(claim, table):
        rows, columns = determine_boundaries(claim)
        number, _, _ = claim
        for i in range(*rows):
            for j in range(*columns):
                location = i, j
                payload = number, location
                increment_cell(payload, table)
                
    def apply_claims(claims, table):
        for claim in claims:
            apply_increments(claim, table)
        
    data = input_to_list(stream, function=str)
    claims = [convert_input(claim) for claim in data]
    table = generate_table(1000)
    apply_claims(claims, table)
    return claims, table

def problem_five(stream):
    _, table = problem_five_six(stream)
    overlapping = sum(1 for row in table for cell in row if len(cell) >= 2)
    return overlapping

def problem_six(stream):
    def get_cell(location, table):
        i, j = location
        return table[i][j]
    
    def check_claim(claim, table):
        rows, columns = determine_boundaries(claim)
        for i in range(*rows):
            for j in range(*columns):
                location = i, j
                if len(get_cell(location, table)) != 1:
                    return False
        return True
    
    claims, table = problem_five_six(stream)
    for claim in claims:
        if check_claim(claim, table):
            return claim

def problem_seven_eight(stream):
    def separate_line(line):
        time, message = line.split(sep='] ')
        processed = time[1:], message
        return processed
        
    def process_line(line):
        time, message = line
        date_length = 11
        hour, minutes = time[date_length:].split(':')
        message = message.split(' ')[1]
        
        return (message, hour, minutes)
    
    def generate_dictionary(lines):
        sleeping_dictionary = {}
        for line in processed:
            message, hour, minutes = line
            if message.startswith('#'):
                number = int(message[1:])
                if number not in sleeping_dictionary:
                    sleeping_dictionary[number] = []
                start = 0 if hour.startswith('23') else int(minutes)
            elif message.startswith('asleep'):
                start = minutes
            elif message.startswith('up'):
                end = minutes
                sleeping_dictionary[number].append((start, end))
    
        return sleeping_dictionary
    
    def expand_ranges(dictionary):
        new_dictionary = {}
        for key, range_list in dictionary.items():
            new_dictionary[key] = []
            for entry in range_list:
                entry = (int(i) for i in entry)
                minutes = [i for i in range(*entry)]
                new_dictionary[key].extend(minutes)
        
        return new_dictionary
        
    def max_sleep_data(dictionary, condition):
        max_guard, max_sleep, max_minute = 0, 0, (0, 0)
        for guard, sleep in dictionary.items():
            total_sleep = len(sleep)
            most_common = max(set(sleep), key=sleep.count) if sleep else 0
            most_common = (sleep.count(most_common), 
                           most_common) if most_common else (0, 0)
            conditions = (total_sleep > max_sleep, 
                          most_common > max_minute)
            if conditions[condition]:
                max_sleep = total_sleep
                max_minute = most_common
                max_guard = guard
        
        return (max_guard, max_sleep, max_minute)
    
    def guard_data_to_solution(guard_data):
        max_guard, _, max_minute = guard_data
        count, minute = max_minute
        return max_guard * minute
                
    data = input_to_list(stream, function=str)
    separated = sorted(separate_line(line) for line in data)
    processed = (process_line(line) for line in separated)
    sleeping_dictionary = generate_dictionary(processed)
    expansions = expand_ranges(sleeping_dictionary)
    condition_7, condition_8 = 0, 1
    guard_7 = max_sleep_data(expansions, condition_7)
    guard_8 = max_sleep_data(expansions, condition_8)
    solutions = (guard_data_to_solution(guard_7),
                 guard_data_to_solution(guard_8))
    return solutions

def problem_nine_and_ten(stream):
    def process_text(text):
        remove = [''.join(i) for i in zip(string.ascii_lowercase,
                  string.ascii_uppercase)]
        removals = [i for i in itertools.chain(remove, 
                                               [i[::-1] for i in remove])]
        text = text[:]
        while True:
            for r in removals:
                text = text.replace(r, '')
            if not any(r in text for r in removals):
                return text
            
    def preprocess_text(text, letter):
        lower, capital = removal = letter.lower(), letter.upper()
        for letter in removal:
            text = text.replace(letter, '')
        return text
    
    #return len(process_text(stream)) #For problem nine.
    
    minimum = math.inf
    letters = string.ascii_lowercase
    for letter in letters:
        preprocessed_text = preprocess_text(stream[:], letter)
        processed_text = process_text(preprocessed_text)
        text_min = len(processed_text)
        if text_min < minimum:
            minimum = text_min
    
    return minimum

def problem_eleven_and_twelve(stream):
    def manhattan_distance(first, second):
        distance = sum(abs(f - s) for f, s in zip(first, second))
        return distance
    
    def nearest_coordinate(cell, coordinates):
        distance_function = functools.partial(manhattan_distance, cell)
        #minimum = min(coordinates, key=distance_function)
        sorted_coordinates = sorted(coordinates, key=distance_function)
        first, second = sorted_coordinates[0:2]
        if distance_function(first) == distance_function(second):
            minimum = None
        else:
            minimum = first
        return minimum
    
    def determine_boundaries(coordinates):
        top = min(coordinates)[0]
        bottom = max(coordinates)[0]
        left = min(coordinates, key=lambda c: c[1])[1]
        right = max(coordinates, key=lambda c: c[1])[1]
        return ((top, bottom), (left, right))
        
    def indexed_table(coordinates):
        boundaries = determine_boundaries(coordinates)
        vertical, horizontal = boundaries
        table = [[(v, h) for h in range(*horizontal)] for v in range(*vertical)]
        return table
    
    def nearest_coordinate_table(coordinates, table):
        output = [[nearest_coordinate(cell, coordinates) for cell in row] for
                  row in table]
        return output
    
    def perimeter_coordinates(table):
        vertical = len(table)
        top = table[0]
        bottom = table[-1]
        left = [table[v][0] for v in range(vertical)]
        right = [table[v][-1] for v in range(vertical)]
        perimeter = set(itertools.chain(top, bottom, left, right))
        
        return perimeter
    
    def conversion(sequence, sep=', '):
        output = tuple(int(i) for i in sequence.split(sep=sep))
        return output
    
    def distance_sum(location, coordinates):
        distances = (manhattan_distance(location, c) for c in coordinates)
        return sum(distances)
    
    coordinates = input_to_list(stream, function=conversion)
#    table = indexed_table(coordinates)
#    nearest_coordinates = nearest_coordinate_table(coordinates, table)
#    perimeter = perimeter_coordinates(nearest_coordinates)
#    viable_coordinates = [c for row in nearest_coordinates for c in row if
#                          c not in perimeter]
#    nearest_counts = ((viable_coordinates.count(i), i) for
#                      i in set(viable_coordinates))
#    maximum = max(nearest_counts)
#    return maximum
    
    vertical, horizontal = determine_boundaries(coordinates)
    limit = 10000
    viable_area = 0
    for i in range(*vertical):
        for j in range(*horizontal):
            location = (i, j)
            if distance_sum(location, coordinates) < limit:
                viable_area += 1
    
    return viable_area

class Worker():
    def __init__(self):
        self.job = None
        self.remaining = math.inf
        self.durations = dict(zip(string.ascii_uppercase, range(61, 87)))
        
    def is_available(self):
        return self.job is None
        
    def accept_job(self, letter):
        self.job = letter
        self.remaining = self.durations[letter]
    
    def tick(self):
        if self.remaining > 0:
            self.remaining -= 1
        
    def job_is_complete(self):
        return self.remaining == 0
    
    def return_job(self):
        return self.job
            
    def reset(self):
        self.job = None
        self.remaining = math.inf        

def problem_thirteen_fourteen(stream):
    def line_to_letter_pair(line, pattern):
        letter_pair = tuple(pattern.findall(line))
        return letter_pair
    
    def determine_requirements(letter_pairs):
        requirements = defaultdict(list)
        for pair in letter_pairs:
            first, second = pair
            requirements[second].append(first)
        return requirements
    
    def problem_thirteen(requirements, text):
        full_set, current, order = set(string.ascii_uppercase), set(), ''
        while True:
            for letter in text:
                required = requirements[letter]
                if all(r in current for r in required):
                    if letter not in order:
                        current.add(letter)
                        order += letter
                        break
            if len(order) == len(full_set):
                return order
            
    def problem_fourteen(requirements, text):
        full_set, current = set(string.ascii_uppercase), set()
        count, total_duration, completed = 5, 0, ''
        workers = [Worker() for i in range(count)]
        available_letters, checked = [], []
        while True:
            for letter in text:
                required = requirements[letter]
                if all(r in current for r in required):
                    if letter not in checked:
                        checked.append(letter)
                        available_letters.append(letter)
            for worker in workers:
                if worker.is_available() and available_letters:
                    worker.accept_job(available_letters.pop(0))
                worker.tick()
                if worker.job_is_complete():
                    letter = worker.return_job()
                    current.add(letter)
                    completed += letter
                    worker.reset()
            total_duration += 1
            if len(completed) == len(full_set):
                return (completed, total_duration)
            
    pattern = re.compile('[Ss]tep (.)')
    text = string.ascii_uppercase
    data = input_to_list(stream, function=str)
    letter_pairs = sorted(line_to_letter_pair(line, pattern) for line in data)
    requirements = determine_requirements(letter_pairs)
    
    #solution = problem_thirteen(requirements, text)
    solution = problem_fourteen(requirements, text)
    return solution

class Tracker():
    def __init__(self, node_count, metadata_count):
        self.initial_nodes = node_count
        self.metadata_count = metadata_count
        self.remaining_nodes = node_count
        
    def __repr__(self):
        message = "Initial: {} Metadata: {} Current: {}"
        packet = self.initial_nodes, self.metadata_count, self.remaining_nodes
        return message.format(*packet)
        
    def is_filled(self):
        return not self.remaining_nodes
        
    def get_initial_nodes(self):
        return self.initial_nodes
        
    def get_metadata_count(self):
        return self.metadata_count
    
    def decrement_remaining(self):
        self.remaining_nodes -= 1

def problem_fifteen(stream):
    def parse_node(sequence):
        node_count, metadata_count, *remainder = sequence
        header = [node_count, metadata_count]
        if not node_count:
            metadata = remainder[:metadata_count]
            sequence = remainder[metadata_count:]
        else:
            metadata = None
            sequence = remainder
        return (header, metadata, sequence)
    
    def decrement_stack_end(node_stack):
        if node_stack:
            node_stack[-1].decrement_remaining()
            
    def contains_nodes(header):
        node_count, metadata_count = header
        return bool(node_count)
    
    def update_stacks(header, metadata, node_stack, metadata_stack):
        if metadata:
            metadata_stack.append(sum(metadata))
        if not contains_nodes(header):
            decrement_stack_end(node_stack)
        else:
            node_stack.append(Tracker(*header))
        
    def process_stack_fifteen(sequence, node_stack, metadata_stack):
        if node_stack:
            metadata = []
            while node_stack[-1].is_filled():
                tracker = node_stack.pop()
                prior_metadata = tracker.get_metadata_count()
                metadata += sequence[:prior_metadata]
                sequence = sequence[prior_metadata:]
                if node_stack:
                    decrement_stack_end(node_stack)
                else:
                    break
        if metadata:
            metadata_stack.append(sum(metadata))
        
        return (metadata_stack, sequence)
    
    def process_stack_sixteen(sequence, node_stack, metadata_stack):
        if node_stack:
            while node_stack[-1].is_filled():
                metadata = 0
                tracker = node_stack.pop()
                initial_nodes = tracker.get_initial_nodes()
                metadata_count = tracker.get_metadata_count()
                node_targets = sequence[:metadata_count]
                sequence = sequence[metadata_count:]
                for node in node_targets:
                    if 1 <= node <= initial_nodes:
                        choice = node - initial_nodes - 1
                        metadata += metadata_stack[choice]
                metadata_stack = metadata_stack[:-initial_nodes]
                metadata_stack.append(metadata)
                if node_stack:
                    decrement_stack_end(node_stack)
                else:
                    break
                
        return (metadata_stack, sequence)

    sequence = input_to_list(stream, delimiter=' ')
    metadata_stack, node_stack = [], []
    pdb.set_trace()
    while sequence:
        header, metadata, sequence = parse_node(sequence)
        update_stacks(header, metadata, node_stack, metadata_stack)
        #metadata_stack, sequence = process_stack_fifteen(sequence, node_stack, metadata_stack)
        metadata_stack, sequence = process_stack_sixteen(sequence, node_stack, metadata_stack)

    solution = sum(metadata_stack)
    
    return solution

message = """9 11 7 3 4 3 3 5 1 7 0 10 2 9 1 6 8 8 5 8 4 9 2 3 3 2 2 1 1 1 7 0 6 3 2 1 3 1 2 1 1 2 1 3 2 3 1 9 0 6 3 1 1 9 2 4 1 2 3 1 1 2 3 3 3 2 1 3 4 1 3 4 1 8 0 10 6 9 7 1 9 7 1 1 1 6 1 1 1 3 1 1 3 1 1 5 0 11 1 1 1 6 8 4 7 2 1 6 2 3 1 1 1 1 1 5 0 10 8 3 7 4 2 6 9 8 5 1 3 3 1 1 2 1 4 3 1 3 6 1 7 0 8 2 8 6 9 8 9 1 8 1 1 2 2 2 2 3 1 5 0 6 7 8 2 5 5 1 1 1 3 1 1 1 6 0 11 6 8 4 5 3 8 2 4 2 6 1 1 1 2 3 2 3 3 4 1 5 2 2 3 6 1 5 0 11 7 2 6 5 5 1 4 8 9 8 3 1 1 2 1 2 1 6 0 6 1 7 1 5 1 6 1 3 1 2 2 3 1 6 0 7 4 1 1 7 4 4 5 3 1 2 3 2 1 3 3 2 4 3 2 6 2 1 4 5 3 5 1 5 0 6 4 3 2 1 2 3 3 1 3 3 1 1 6 0 7 3 4 6 1 9 3 8 1 2 1 3 3 3 1 5 0 6 5 1 1 1 9 1 3 1 1 1 3 2 2 5 3 4 3 6 1 8 0 9 5 3 2 1 7 8 9 1 9 1 1 3 2 2 3 3 2 1 6 0 9 2 2 1 1 7 3 2 4 5 1 1 3 1 3 2 1 6 0 8 6 7 6 1 5 9 8 8 3 1 3 1 2 3 2 4 2 2 1 5 3 4 1 9 0 9 6 6 1 4 2 1 1 6 8 2 1 1 2 3 3 1 2 2 1 7 0 11 4 4 2 2 9 4 3 3 3 1 5 3 2 1 2 1 2 3 1 8 0 10 4 1 9 7 9 8 3 8 5 3 1 1 3 3 1 2 2 1 1 1 3 2 3 4 1 9 0 8 6 1 7 1 3 1 2 9 1 1 2 3 2 3 2 1 2 1 9 0 8 1 7 2 8 9 6 5 1 2 2 2 1 1 1 2 3 3 1 9 0 6 1 9 6 8 1 4 1 3 1 3 3 2 1 2 3 3 2 1 2 3 1 5 4 4 5 5 3 6 1 6 0 10 4 9 9 1 9 5 9 3 1 7 1 1 2 1 1 2 1 6 0 6 8 1 9 3 5 5 2 3 1 2 3 2 1 5 0 11 1 1 4 8 6 9 1 3 7 2 5 1 3 1 2 1 1 3 4 4 4 3 3 5 1 5 0 6 1 1 4 5 4 6 1 1 1 2 3 1 7 0 7 7 1 1 2 9 4 1 2 1 1 1 3 2 2 1 8 0 8 1 1 1 7 4 1 2 9 2 2 1 2 2 3 3 2 2 2 5 5 1 3 5 1 5 0 7 8 2 9 5 1 1 4 3 3 2 1 3 1 8 0 6 7 8 3 8 1 7 2 1 1 2 1 1 3 3 1 5 0 11 7 3 8 3 8 1 8 3 1 7 4 1 1 2 1 3 4 3 4 4 3 3 5 1 8 0 9 8 3 7 7 1 7 3 9 7 2 3 1 2 1 2 1 3 1 8 0 11 7 5 3 2 3 7 7 1 1 4 8 2 3 1 1 2 3 1 1 1 7 0 11 6 2 4 4 3 1 6 3 9 5 7 1 3 1 1 1 2 1 5 1 1 5 1 3 5 1 9 0 6 5 4 7 1 8 1 3 2 1 2 1 1 3 1 1 1 7 0 11 1 8 1 8 1 5 1 1 2 6 9 2 1 1 3 2 3 1 1 5 0 10 6 1 7 2 1 8 2 1 2 2 2 1 1 3 2 1 2 4 3 2 5 2 5 5 7 5 4 3 4 1 8 0 7 3 2 5 1 5 9 5 3 3 2 3 3 3 1 1 1 5 0 10 7 1 9 5 7 8 2 7 6 9 2 3 2 1 2 1 6 0 9 5 9 1 5 1 5 3 2 1 3 1 3 1 1 2 3 3 2 2 3 4 1 6 0 11 2 6 5 8 1 6 6 1 1 8 5 1 1 1 1 1 1 1 6 0 11 4 9 2 7 1 9 2 7 2 7 1 1 3 3 3 1 2 1 9 0 10 3 1 8 6 3 2 4 6 2 2 2 3 1 3 2 1 1 2 2 1 1 3 2 3 6 1 8 0 6 7 1 8 7 3 1 2 1 1 2 1 2 2 2 1 5 0 10 8 2 1 7 6 9 3 7 4 7 1 2 2 1 3 1 9 0 6 9 2 5 1 1 1 3 1 1 3 2 3 2 2 2 3 5 3 1 2 2 3 5 1 8 0 6 8 8 1 6 1 9 2 2 3 1 1 3 3 2 1 8 0 6 7 7 2 1 1 7 2 1 1 1 1 3 2 2 1 5 0 6 6 8 1 7 7 7 3 1 3 1 1 3 5 2 3 2 3 4 1 7 0 7 1 4 3 1 7 8 8 1 1 2 3 1 2 2 1 5 0 11 1 3 9 8 6 1 5 2 1 7 7 1 1 3 2 3 1 5 0 10 4 2 7 1 1 4 9 8 6 6 3 1 1 1 3 3 1 5 3 4 5 7 3 5 4 3 5 1 6 0 10 3 3 6 7 1 6 2 8 4 7 3 1 1 1 1 1 1 7 0 11 3 8 4 1 1 3 8 5 5 3 3 3 1 1 2 3 1 1 1 7 0 8 5 2 1 8 1 9 1 4 3 2 1 2 2 3 1 2 1 3 2 3 3 7 1 9 0 6 4 5 1 1 7 4 2 3 3 1 2 3 1 1 1 1 5 0 6 9 1 5 4 6 1 1 1 3 1 2 1 9 0 10 2 1 2 1 9 9 6 6 8 8 2 2 1 1 3 1 3 1 1 3 1 3 2 5 4 3 3 4 1 8 0 11 6 7 6 9 2 1 6 4 5 6 2 2 1 1 2 2 2 1 2 1 8 0 7 2 4 9 3 4 9 1 1 2 1 3 1 1 3 1 1 6 0 9 5 2 8 4 2 1 5 9 7 1 1 2 3 1 1 2 3 3 4 3 4 1 8 0 9 9 9 8 1 9 8 8 8 6 3 1 2 1 2 2 1 3 1 8 0 7 6 7 2 3 6 1 1 3 1 2 1 2 2 1 2 1 7 0 10 1 1 3 5 4 8 6 2 4 5 3 1 1 1 1 3 3 5 1 5 4 3 7 1 6 0 7 4 2 9 5 4 1 3 1 1 1 2 1 2 1 9 0 7 6 8 7 6 8 1 7 1 3 1 2 1 2 1 1 3 1 7 0 9 1 9 1 2 6 1 7 7 4 2 1 1 2 2 1 2 4 1 2 2 1 4 2 6 7 6 2 4 3 3 5 1 8 0 10 1 5 4 4 2 9 1 5 7 4 3 2 1 3 1 2 2 2 1 9 0 6 1 8 5 6 1 9 3 3 2 3 1 2 2 2 3 1 8 0 9 7 7 9 9 5 3 8 1 3 3 1 3 2 2 2 1 1 4 5 3 3 2 3 7 1 8 0 9 9 7 1 5 1 3 1 8 3 3 1 1 1 1 2 3 1 1 6 0 9 1 2 6 9 2 3 4 9 9 1 3 3 3 2 1 1 5 0 8 7 4 2 5 6 1 4 3 1 3 3 2 2 2 5 3 3 1 4 1 3 4 1 9 0 6 1 4 7 7 2 9 2 1 2 2 3 3 1 1 3 1 9 0 9 8 3 5 2 1 6 2 4 7 1 2 1 2 3 2 2 3 1 1 9 0 11 5 1 7 1 4 7 2 2 1 9 1 3 3 3 1 1 3 2 2 1 2 3 5 5 3 6 1 6 0 7 2 1 7 1 3 7 4 1 1 1 1 1 2 1 6 0 11 7 9 5 9 8 3 6 5 1 5 1 3 1 3 1 1 1 1 8 0 6 2 8 9 1 5 1 1 1 3 2 1 3 2 2 1 5 5 2 3 5 2 2 3 5 5 3 5 1 6 0 8 1 9 1 2 7 8 3 3 1 2 3 3 1 1 1 6 0 8 6 7 4 8 7 1 3 1 3 1 3 1 2 2 1 7 0 8 7 2 8 6 3 1 1 3 2 1 1 1 3 2 2 5 3 5 2 1 3 7 1 8 0 10 6 2 1 1 6 6 1 7 2 6 1 3 1 2 2 1 1 3 1 5 0 10 7 9 6 5 3 6 3 1 8 6 1 2 1 2 1 1 8 0 9 1 4 5 6 3 2 8 1 3 1 1 2 3 3 1 1 1 3 1 2 2 1 4 1 3 5 1 7 0 10 8 3 5 9 1 2 9 1 4 3 2 3 3 1 2 3 1 1 9 0 8 4 1 3 5 8 2 5 2 3 2 3 1 1 1 3 2 2 1 9 0 9 1 5 7 4 1 3 2 5 2 2 2 1 1 3 2 3 1 1 4 5 3 2 1 3 4 1 8 0 8 7 3 1 1 3 6 9 2 2 1 1 2 2 2 1 1 1 9 0 11 1 9 5 1 5 5 1 1 5 9 3 1 3 1 1 1 1 3 2 3 1 6 0 11 5 1 8 8 3 6 2 6 2 8 6 2 3 2 2 1 2 3 1 3 4 3 5 1 8 0 9 1 1 9 7 6 5 4 1 9 2 2 2 2 1 1 1 1 1 8 0 10 4 5 8 1 3 3 4 7 3 7 1 3 2 2 3 3 1 1 1 5 0 9 3 8 9 7 4 8 8 1 1 3 1 1 2 3 2 1 2 2 1 4 7 3 4 4 4 4 3 6 2 5 5 3 4 1 9 0 7 8 6 3 8 3 4 1 1 1 2 2 1 2 2 3 3 1 8 0 9 9 2 4 7 1 4 2 3 3 1 3 1 3 1 3 2 1 1 6 0 11 1 6 6 7 2 9 9 9 2 4 7 3 2 1 1 3 3 1 4 3 5 3 4 1 9 0 9 2 1 1 1 9 6 2 3 4 1 3 1 1 2 3 1 2 3 1 6 0 10 3 7 8 3 3 1 6 1 5 4 1 2 3 1 3 2 1 9 0 7 6 6 2 3 4 9 1 1 3 2 2 1 1 1 1 3 1 5 5 5 3 7 1 8 0 8 7 8 1 8 6 5 1 9 3 2 3 3 2 2 1 3 1 9 0 8 4 6 2 1 7 6 4 1 2 2 2 3 2 1 2 3 1 1 6 0 7 1 1 5 5 6 3 3 3 3 2 3 1 3 3 2 3 4 5 2 5 3 5 1 8 0 11 9 5 7 8 1 1 7 5 6 4 1 2 3 3 2 2 1 1 2 1 9 0 7 9 5 4 1 8 5 7 3 1 1 1 3 1 3 1 1 1 9 0 9 9 7 1 2 3 8 4 2 9 3 3 2 3 3 1 3 1 3 2 2 2 1 2 3 5 1 7 0 7 8 4 7 1 9 4 6 1 2 2 1 3 2 2 1 6 0 6 6 5 9 1 6 9 1 3 2 2 1 2 1 6 0 7 4 1 2 5 5 5 3 3 1 2 3 2 1 1 3 3 4 2 6 5 7 3 7 5 5 3 7 1 8 0 6 5 3 1 7 4 3 2 1 1 3 2 3 1 3 1 8 0 8 1 2 6 5 5 4 1 4 1 1 3 3 1 1 1 1 1 9 0 9 1 9 3 6 7 5 9 3 8 3 1 2 1 1 2 1 2 1 2 3 3 3 2 2 1 3 7 1 6 0 7 6 2 2 6 7 1 3 3 2 1 2 1 1 1 5 0 10 3 9 7 3 9 1 2 7 6 5 1 1 1 2 1 1 5 0 11 8 1 1 6 3 4 7 2 1 5 2 2 1 1 2 3 1 1 4 1 5 2 1 3 7 1 9 0 9 7 5 8 5 4 8 6 1 8 3 1 2 1 1 2 1 3 3 1 5 0 8 7 3 1 1 2 1 8 6 3 3 1 2 1 1 8 0 10 5 4 9 4 2 4 9 1 1 8 3 2 1 2 1 2 3 3 2 4 4 4 1 1 2 3 4 1 6 0 8 6 7 7 3 8 1 7 7 1 3 3 3 2 2 1 7 0 8 7 7 8 1 5 8 6 3 1 1 2 1 2 1 3 1 9 0 10 3 1 5 8 1 4 1 1 6 1 3 1 2 1 1 2 2 2 1 1 3 1 5 3 4 1 7 0 6 1 9 2 9 9 9 2 3 2 1 1 1 2 1 9 0 11 1 5 2 3 4 1 7 9 8 7 1 1 2 2 2 1 1 3 1 2 1 8 0 10 8 8 1 1 7 8 3 7 9 3 1 1 1 1 2 1 3 3 3 2 2 5 4 4 7 2 7 4 4 3 4 1 5 0 9 4 1 2 6 2 5 9 7 4 1 1 3 3 2 1 5 0 10 1 1 9 5 4 2 2 7 1 5 2 1 1 3 1 1 5 0 6 8 6 6 1 1 6 1 3 1 2 1 5 2 1 5 3 4 1 7 0 6 1 7 6 5 4 5 1 2 2 1 2 3 2 1 8 0 7 8 1 7 8 2 7 5 2 2 1 2 2 2 3 3 1 5 0 11 9 1 7 8 2 3 7 4 9 1 7 2 1 3 2 3 3 2 5 1 3 4 1 5 0 7 5 1 3 2 3 7 1 2 2 1 3 2 1 8 0 7 1 4 5 1 2 7 5 3 1 1 2 1 1 1 1 1 7 0 10 6 6 1 5 7 7 4 1 4 3 1 1 1 2 2 1 3 1 5 4 3 3 7 1 6 0 8 5 7 8 4 1 5 7 1 1 3 1 2 1 2 1 5 0 8 5 6 4 1 9 1 4 8 1 2 2 1 1 1 8 0 8 1 4 9 3 3 7 9 7 2 3 3 1 1 1 2 2 2 5 1 4 3 4 1 3 4 1 5 5 5 3 7 1 7 0 8 4 7 5 1 1 8 5 2 2 1 1 3 1 2 1 1 5 0 7 9 3 9 1 1 4 2 3 1 2 1 2 1 5 0 7 1 2 4 5 2 8 6 1 3 1 1 2 5 5 4 1 1 3 2 3 5 1 7 0 8 9 7 9 9 6 2 9 1 3 1 1 2 3 2 1 1 5 0 7 1 2 5 4 9 1 5 2 3 1 1 2 1 7 0 6 9 7 1 3 3 8 1 1 3 3 2 1 3 2 5 2 1 5 3 6 1 7 0 9 1 7 4 8 1 3 4 8 3 3 2 2 1 3 2 1 1 5 0 9 2 5 7 9 6 1 2 4 5 1 2 2 1 3 1 8 0 9 3 7 1 2 1 7 9 9 8 1 3 2 3 2 1 2 3 1 3 4 5 3 3 3 5 1 5 0 11 6 6 1 4 9 6 8 7 3 1 7 3 1 3 2 1 1 8 0 9 9 3 8 8 5 1 6 5 2 1 1 2 1 3 2 2 1 1 9 0 6 1 6 1 6 4 3 1 1 1 3 3 2 1 3 3 2 1 3 2 3 3 5 1 5 0 8 2 2 1 5 4 5 6 7 1 1 1 3 3 1 7 0 9 9 1 6 6 9 8 7 3 7 1 3 2 3 1 2 1 1 7 0 6 5 9 9 4 2 1 3 1 2 3 2 1 2 2 5 4 1 3 3 6 2 1 2 5 4 3 7 1 7 0 9 8 9 2 1 3 6 1 4 8 3 1 3 3 1 1 2 1 8 0 8 4 5 1 1 2 2 7 6 1 3 3 3 3 1 2 1 1 6 0 6 7 7 1 3 6 8 2 1 2 1 2 2 3 2 4 1 5 2 3 3 5 1 8 0 8 3 8 1 7 1 5 1 8 2 3 2 3 1 2 2 1 1 8 0 8 3 4 8 8 1 2 1 8 1 1 1 3 1 3 1 1 1 5 0 9 1 7 7 9 2 5 6 5 2 1 1 3 1 1 1 3 5 2 5 3 5 1 6 0 7 6 6 1 9 3 9 2 3 1 3 1 1 2 1 7 0 11 6 7 9 1 9 8 4 4 3 1 9 2 1 1 3 1 3 3 1 8 0 10 4 7 4 7 9 1 7 1 3 1 2 1 3 1 3 1 3 2 3 5 2 4 4 3 5 1 5 0 10 1 5 1 6 3 6 1 6 1 9 2 1 1 3 2 1 9 0 6 9 3 4 7 1 1 1 2 3 2 1 1 1 2 1 1 9 0 8 6 4 7 7 1 9 1 6 1 2 3 2 1 2 1 1 1 5 1 4 2 3 3 4 1 6 0 11 8 2 1 8 5 4 3 5 8 1 8 1 2 3 2 2 1 1 7 0 7 1 4 5 3 8 8 4 3 1 2 2 3 2 1 1 8 0 8 8 2 1 4 2 7 7 9 1 3 2 1 2 2 3 2 1 4 2 4 2 4 1 1 5 5 3 6 1 7 0 6 8 1 2 8 2 1 2 2 3 3 1 1 3 1 6 0 10 8 6 6 6 1 6 2 6 5 3 3 1 1 2 1 1 1 6 0 7 2 4 3 6 8 1 5 1 1 3 2 2 2 5 3 3 3 1 3 3 6 1 8 0 6 7 1 1 3 8 8 1 1 1 1 3 3 3 1 1 9 0 9 5 1 2 8 3 8 5 3 3 3 1 3 2 2 2 3 1 1 1 6 0 10 7 9 2 9 9 4 2 1 9 2 1 1 1 1 3 2 1 2 3 2 2 5 3 5 1 5 0 10 5 7 8 9 2 1 6 8 5 3 3 1 2 1 3 1 7 0 9 3 1 8 6 3 7 3 1 6 1 1 1 1 2 1 1 1 9 0 8 7 6 5 1 3 2 6 4 3 1 3 3 1 3 1 2 3 3 1 2 5 3 3 7 1 7 0 11 4 4 3 4 9 8 5 5 9 5 1 1 2 2 2 2 1 2 1 8 0 10 8 9 6 1 6 9 4 2 3 3 1 3 1 3 2 1 3 1 1 9 0 10 9 7 1 7 9 4 5 1 5 7 1 2 1 2 2 2 1 3 3 4 1 4 3 4 1 3 3 5 1 8 0 11 9 5 9 1 9 4 2 5 1 2 2 2 1 3 3 2 2 2 1 1 9 0 6 3 9 7 9 6 1 3 1 2 2 3 2 1 1 1 1 9 0 11 3 7 4 3 3 5 5 4 2 1 5 2 1 3 3 1 3 2 2 1 5 1 3 1 5 4 6 5 1 3 3 4 6 3 5 3 3 7 1 6 0 8 1 2 5 3 8 6 2 1 1 3 1 2 1 3 1 6 0 7 1 4 7 3 5 6 3 1 2 2 1 2 2 1 7 0 10 2 9 5 8 1 1 7 6 4 8 2 1 2 1 1 1 1 1 3 2 4 5 3 5 3 4 1 6 0 6 7 5 1 3 1 7 3 1 1 2 1 3 1 7 0 11 4 4 5 5 6 8 6 1 1 3 7 1 2 3 2 1 1 3 1 7 0 7 8 8 1 1 7 2 9 2 1 2 2 3 2 1 4 3 4 5 3 5 1 7 0 6 1 1 5 2 8 7 1 1 3 1 3 2 3 1 8 0 10 1 5 7 2 6 2 3 7 3 8 1 1 2 1 2 1 3 3 1 8 0 6 1 1 3 8 1 5 1 1 1 2 2 2 1 3 2 5 2 4 1 3 4 1 8 0 9 9 4 6 4 7 1 4 1 4 3 2 1 2 3 3 2 3 1 5 0 7 1 2 1 1 9 3 1 1 3 1 1 2 1 8 0 9 5 1 7 1 6 3 6 2 5 1 3 3 1 1 3 1 3 3 5 3 1 3 5 1 7 0 9 8 6 4 3 5 3 9 9 1 1 1 3 2 3 1 1 1 5 0 7 3 3 4 5 6 1 6 1 1 3 3 2 1 7 0 8 1 5 2 2 2 6 6 1 1 3 2 1 2 2 2 3 2 3 5 4 2 2 7 5 3 3 4 1 7 0 6 2 9 1 1 1 6 2 1 1 1 1 1 2 1 9 0 7 5 6 7 1 2 6 1 1 1 1 2 1 1 3 3 3 1 6 0 8 3 1 5 8 9 6 9 4 1 3 3 1 3 1 2 1 1 4 3 6 1 5 0 11 1 1 6 5 7 4 2 9 2 8 7 1 1 1 1 3 1 7 0 6 1 7 1 3 5 7 2 1 1 1 1 2 3 1 9 0 11 2 3 9 1 5 4 2 9 1 6 5 3 1 2 2 3 3 1 2 2 2 5 5 5 5 2 3 7 1 8 0 10 9 8 5 3 8 7 9 1 1 1 1 3 2 3 1 1 2 2 1 5 0 6 2 1 4 7 7 2 3 2 1 1 2 1 8 0 9 1 8 5 9 4 4 7 3 1 1 2 2 1 1 3 2 1 1 5 5 2 4 5 5 3 7 1 9 0 8 4 1 1 4 1 4 2 8 1 2 2 1 1 1 3 3 3 1 6 0 11 9 3 1 5 8 4 7 2 2 1 4 2 3 3 1 2 3 1 7 0 6 1 1 4 4 7 4 2 3 2 1 3 3 1 2 1 1 1 3 4 5 3 4 1 8 0 9 9 9 9 4 8 1 1 3 1 2 1 2 1 2 3 2 3 1 8 0 10 6 1 2 8 4 9 7 1 1 5 2 1 2 1 2 3 2 2 1 9 0 6 2 2 4 1 3 3 3 1 1 2 3 1 3 3 3 4 5 2 2 6 3 7 5 3 3 4 1 8 0 7 9 5 3 6 1 3 2 1 2 2 1 1 2 3 1 1 8 0 8 5 4 1 3 1 4 9 7 1 2 3 3 3 1 3 2 1 6 0 8 6 1 9 9 4 9 2 1 2 1 2 1 2 1 2 2 4 1 3 7 1 8 0 10 9 1 5 4 2 9 5 3 1 6 3 1 3 2 2 3 1 1 1 7 0 10 5 6 3 3 5 7 4 1 3 1 2 1 3 1 2 1 3 1 6 0 11 5 3 7 1 2 1 8 4 9 1 1 3 1 1 2 3 1 2 1 2 4 4 2 4 3 4 1 9 0 9 5 3 7 4 9 4 5 7 1 1 1 1 2 2 2 2 3 1 1 6 0 10 6 8 2 7 2 4 4 1 2 2 2 3 3 1 1 1 1 7 0 6 6 1 1 9 5 7 2 1 2 2 3 3 3 5 4 3 2 3 4 1 5 0 11 5 1 9 6 2 3 5 9 7 2 1 1 1 3 1 3 1 8 0 11 1 9 9 6 5 7 9 4 6 4 9 1 3 1 1 1 2 1 1 1 8 0 7 1 1 7 8 9 1 1 1 2 3 2 1 3 2 3 2 3 3 4 3 7 1 9 0 8 8 4 2 3 5 6 1 6 1 3 1 1 2 1 3 2 2 1 7 0 9 2 9 7 6 1 6 3 9 7 2 2 3 1 1 1 3 1 9 0 11 3 8 8 7 7 8 4 4 9 4 1 1 1 2 2 2 1 1 1 3 2 4 5 2 2 5 3 7 3 1 5 5 3 6 1 9 0 6 1 2 1 2 4 4 2 2 2 1 1 3 1 3 2 1 7 0 8 4 2 1 9 3 4 1 7 2 3 1 1 3 2 3 1 8 0 7 1 8 1 6 3 9 8 2 1 2 1 2 1 2 3 2 5 3 2 3 5 3 5 1 6 0 6 1 1 3 9 8 4 1 1 3 3 1 1 1 8 0 11 1 1 7 7 3 8 1 2 7 4 6 3 1 3 3 1 2 2 2 1 6 0 8 7 7 5 3 5 2 2 1 2 3 2 3 3 1 3 4 2 5 5 3 7 1 9 0 8 8 4 6 7 8 2 1 3 2 1 1 1 3 2 2 1 2 1 6 0 10 2 1 1 2 5 1 1 6 3 6 1 2 3 2 1 1 1 7 0 6 1 4 6 4 3 9 1 3 1 3 3 1 2 4 2 3 4 3 1 5 3 6 1 6 0 6 8 2 8 8 1 7 1 1 2 1 1 1 1 6 0 9 4 6 9 2 7 1 3 2 9 2 2 3 2 1 1 1 8 0 6 2 2 1 4 1 1 1 2 1 2 2 3 2 2 4 2 5 1 1 1 3 4 1 8 0 6 5 8 1 1 1 8 1 3 2 2 1 2 2 2 1 9 0 11 9 1 6 1 7 8 3 3 5 2 8 1 2 3 2 1 1 2 2 2 1 6 0 9 6 2 2 3 4 6 1 2 2 3 2 1 2 1 3 2 1 5 4 2 1 5 1 3 5 3 3 6 1 6 0 10 7 8 1 4 1 7 6 6 7 7 1 1 1 2 2 1 1 5 0 7 8 8 1 1 3 6 7 1 3 3 3 1 1 6 0 8 3 6 1 1 2 3 4 8 1 1 1 3 3 3 5 3 2 1 2 5 3 4 1 9 0 9 6 2 2 1 4 1 2 3 1 3 3 2 2 2 2 1 1 3 1 5 0 6 7 3 5 8 1 3 2 1 2 1 1 1 7 0 11 1 5 2 3 9 1 1 9 9 5 3 1 3 2 1 2 3 3 5 4 1 1 3 7 1 5 0 10 4 6 4 9 7 9 1 8 8 7 1 1 1 3 1 1 5 0 11 1 1 4 5 1 8 4 1 3 6 6 2 1 3 3 1 1 7 0 7 8 6 1 3 1 6 5 1 2 2 1 2 1 1 4 1 3 4 1 3 1 3 7 1 8 0 6 9 9 7 6 1 6 1 2 1 3 2 1 3 3 1 7 0 11 7 1 1 7 6 2 6 1 3 2 1 3 2 1 1 3 2 3 1 6 0 10 9 7 5 3 7 3 2 1 1 5 3 1 2 1 1 3 2 5 3 5 1 1 3 3 6 1 9 0 9 1 7 6 5 7 4 3 5 3 2 1 2 1 1 2 3 1 3 1 8 0 6 8 7 1 7 3 7 3 3 1 1 2 2 1 3 1 9 0 7 7 2 2 1 4 1 3 1 2 2 3 2 1 3 1 1 3 3 2 4 4 3 5 1 3 5 3 3 7 1 7 0 7 7 1 8 9 5 8 1 1 2 1 3 1 3 2 1 5 0 11 6 3 7 2 1 7 8 5 4 7 8 2 1 1 3 1 1 5 0 8 5 2 4 1 1 2 1 9 3 2 1 1 2 2 1 3 3 2 3 5 3 6 1 7 0 6 5 2 1 4 1 7 3 2 1 2 1 3 1 1 7 0 7 6 6 3 1 4 8 7 1 2 3 3 1 3 3 1 7 0 6 2 4 1 7 4 9 2 3 3 1 3 1 3 3 1 5 3 2 5 3 6 1 5 0 8 5 4 1 4 1 6 6 2 1 3 3 1 2 1 6 0 7 7 3 7 3 1 5 1 1 3 1 3 2 2 1 8 0 9 1 2 3 5 7 1 5 7 4 3 1 1 1 1 2 3 1 3 4 3 3 3 5 3 6 1 5 0 10 1 6 6 1 1 9 8 5 8 8 3 1 2 1 3 1 5 0 11 9 6 1 7 1 8 1 3 6 6 3 1 2 3 1 1 1 7 0 8 2 5 1 6 1 8 6 2 3 2 2 3 1 1 1 1 5 1 1 2 4 3 5 1 9 0 6 5 1 8 2 4 6 2 2 1 1 2 1 1 1 1 1 8 0 11 7 2 7 6 8 3 1 1 8 1 7 1 2 3 1 1 1 2 1 1 6 0 10 3 1 2 8 2 1 4 6 4 2 3 2 1 2 2 1 3 3 1 3 1 4 4 1 2 2 1 7 2 5 5 3 4 1 7 0 11 3 1 4 2 1 2 7 7 9 1 2 1 2 2 2 3 3 1 1 7 0 10 3 4 1 1 7 9 8 6 9 8 3 1 1 2 3 1 1 1 8 0 9 8 7 6 4 1 1 7 1 1 3 3 1 1 2 2 3 1 3 5 5 5 3 6 1 5 0 9 4 1 6 8 5 8 2 4 1 2 2 1 1 2 1 5 0 10 8 5 2 1 4 5 7 1 4 1 1 3 2 1 1 1 7 0 7 7 3 8 2 7 1 1 3 1 2 2 1 3 2 1 1 4 1 5 1 3 5 1 6 0 9 6 1 4 7 4 8 8 5 7 3 1 1 2 1 3 1 5 0 6 8 1 4 3 4 9 1 1 3 3 2 1 7 0 7 5 6 3 1 1 5 9 2 3 2 1 3 1 2 1 3 3 1 1 3 5 1 8 0 7 4 5 1 3 9 2 6 1 3 2 1 1 1 3 1 1 7 0 9 1 7 5 6 1 4 7 2 3 1 3 1 2 3 2 1 1 5 0 8 9 8 1 1 3 2 1 1 3 1 2 1 2 3 4 1 2 4 3 5 1 6 0 7 6 2 2 3 1 5 9 1 2 1 1 1 2 1 9 0 6 6 5 1 7 1 7 1 1 1 3 1 3 2 3 3 1 8 0 7 7 8 1 1 5 9 7 1 1 3 1 2 3 2 2 5 3 5 4 1 2 3 2 4 2 5 5 3 7 1 6 0 6 5 6 9 5 1 1 1 1 1 2 3 1 1 9 0 9 2 6 8 1 1 5 8 1 2 3 2 2 2 3 3 1 2 1 1 8 0 6 5 8 7 4 1 1 1 3 1 2 1 2 3 1 3 4 1 1 2 5 1 3 5 1 5 0 8 4 8 3 7 7 3 1 1 1 2 2 2 1 1 7 0 11 2 8 2 7 9 4 2 3 7 1 1 2 3 2 3 1 1 2 1 8 0 7 2 8 4 7 7 5 1 2 3 3 2 2 3 1 3 4 3 1 1 1 3 5 1 6 0 11 7 1 5 9 6 1 7 6 5 3 7 3 2 1 1 3 2 1 9 0 10 6 1 9 5 8 6 1 9 7 9 1 2 3 2 1 3 2 2 3 1 9 0 8 8 3 1 5 4 8 4 8 1 2 1 3 1 3 2 1 1 1 1 3 1 4 3 5 1 8 0 8 8 1 1 5 7 8 7 8 2 3 2 1 3 1 1 1 1 8 0 7 3 3 8 8 6 1 2 3 1 3 1 2 1 3 1 1 9 0 8 7 2 2 1 5 5 5 4 3 2 3 3 2 3 2 1 1 5 2 5 1 4 3 7 1 9 0 6 1 9 4 8 2 7 1 2 1 3 2 1 2 2 3 1 6 0 11 1 8 1 2 9 1 7 9 4 1 1 3 1 3 1 2 1 1 6 0 6 1 4 7 3 5 5 2 2 2 1 1 1 5 4 4 5 2 2 4 2 6 6 2 2 5 3 3 5 1 5 0 8 1 1 5 4 4 1 5 5 1 3 2 1 3 1 9 0 11 4 9 1 6 3 6 8 9 8 5 8 3 1 3 1 2 3 1 3 3 1 8 0 8 1 1 6 6 6 5 9 3 1 1 1 1 2 2 3 2 1 4 2 3 2 3 4 1 6 0 7 6 6 3 9 1 1 5 1 1 3 1 1 1 1 6 0 10 9 9 3 1 1 3 9 6 5 4 1 1 2 2 1 3 1 9 0 10 3 8 9 6 6 9 6 8 1 9 3 1 1 1 2 3 1 1 1 3 5 1 5 3 5 1 9 0 7 1 9 6 6 1 6 3 1 3 2 1 1 2 1 2 2 1 6 0 6 1 2 9 6 3 6 1 2 3 1 1 3 1 7 0 7 4 3 9 1 8 1 1 1 1 3 2 1 3 2 4 3 1 2 4 3 4 1 9 0 7 7 4 8 1 8 5 1 1 3 1 1 2 2 1 2 3 1 9 0 11 5 6 8 7 1 8 1 9 1 7 1 1 2 3 1 2 1 1 3 2 1 7 0 6 7 1 7 9 1 2 1 3 2 2 1 1 1 5 2 4 1 3 7 1 6 0 7 2 5 2 1 4 3 5 2 3 1 2 2 3 1 7 0 10 4 7 5 8 2 7 1 6 2 1 1 3 1 1 3 2 1 1 5 0 7 5 7 1 7 3 8 1 3 1 3 1 1 3 3 1 2 2 2 4 6 4 3 5 4 3 5 1 6 0 6 4 1 9 6 1 9 3 2 3 1 2 1 1 6 0 9 1 8 8 2 5 6 1 6 7 2 1 1 1 3 3 1 6 0 10 1 1 6 8 5 2 6 3 3 6 1 3 2 3 1 2 4 1 1 1 4 3 6 1 6 0 6 4 2 1 1 2 6 2 2 2 1 2 1 1 9 0 9 8 3 1 8 9 1 8 6 6 3 1 2 1 2 1 3 3 3 1 8 0 6 3 7 9 2 1 9 1 1 2 1 1 2 1 3 1 3 1 4 2 5 3 6 1 6 0 6 1 4 4 3 9 6 1 3 3 3 1 2 1 9 0 11 9 3 9 1 3 6 5 6 4 8 1 3 3 3 1 1 2 3 3 1 1 7 0 11 5 7 5 2 2 2 1 4 5 6 8 1 2 2 2 1 2 3 5 2 3 1 2 5 3 5 1 8 0 7 1 7 5 5 6 6 7 1 2 2 1 3 2 2 3 1 5 0 6 3 1 9 3 5 2 3 2 1 3 1 1 7 0 8 1 8 6 5 7 1 1 2 3 3 3 2 1 3 1 4 1 5 5 2 3 6 1 9 0 8 1 8 1 7 3 8 7 8 3 2 1 2 1 2 3 1 3 1 9 0 9 5 6 1 2 8 6 6 9 7 1 3 3 3 1 2 3 1 3 1 8 0 6 5 7 1 7 1 2 2 1 1 3 3 3 2 1 3 3 1 5 3 2 5 1 2 2 4 3 3 7 1 8 0 7 1 9 7 1 3 3 5 3 3 1 1 3 2 3 2 1 8 0 11 9 5 5 2 8 2 5 7 5 1 5 3 3 1 3 1 2 1 1 1 5 0 11 5 9 2 6 4 8 1 7 4 4 3 2 2 1 3 1 2 1 4 4 3 5 4 3 5 1 5 0 8 9 1 6 5 4 6 9 8 2 1 1 2 2 1 9 0 9 4 6 3 7 7 5 1 4 4 1 1 3 1 1 2 1 1 1 1 9 0 7 1 3 1 2 1 9 9 2 2 3 1 1 3 1 3 1 5 4 5 1 3 3 7 1 5 0 8 2 1 5 1 7 7 3 7 1 2 3 2 1 1 7 0 6 4 2 5 1 6 7 1 3 1 2 2 1 1 1 9 0 10 1 5 6 3 2 7 1 4 5 5 3 1 1 1 3 2 2 2 2 5 1 3 5 1 2 5 3 7 1 5 0 9 8 1 8 2 7 5 9 3 6 3 2 2 1 1 1 7 0 11 1 2 5 7 3 5 5 8 1 2 7 1 3 2 3 3 1 3 1 8 0 8 3 2 8 6 8 6 9 1 1 1 3 1 2 1 1 1 5 4 1 3 4 1 2 3 4 4 5 3 3 4 1 5 0 8 9 4 7 1 6 4 7 3 1 1 2 1 2 1 7 0 10 2 3 1 8 3 7 6 2 7 6 2 1 2 1 2 1 2 1 5 0 11 6 1 1 1 4 5 1 2 6 5 2 2 3 3 1 2 3 3 3 1 3 7 1 5 0 8 4 3 1 2 9 1 8 6 3 1 2 2 1 1 8 0 7 1 6 5 1 3 8 5 1 1 1 2 1 1 2 1 1 8 0 8 7 5 5 6 6 7 1 6 2 1 1 2 1 2 1 2 3 3 4 4 1 1 3 3 7 1 9 0 10 4 3 6 1 1 4 3 8 7 6 1 1 1 2 2 3 1 3 3 1 5 0 7 1 6 8 2 7 2 6 2 1 2 2 1 1 7 0 6 8 8 8 6 1 2 3 1 1 3 2 1 3 2 3 1 4 5 5 3 3 4 1 5 0 6 5 3 1 7 3 1 1 3 1 2 2 1 7 0 6 8 4 5 1 1 9 3 2 2 1 2 2 2 1 8 0 9 5 3 8 6 1 6 8 5 4 2 2 2 1 3 3 3 3 2 1 1 3 3 6 1 8 0 10 8 7 1 9 1 1 5 8 6 6 2 3 2 1 3 3 3 3 1 9 0 9 6 1 1 3 3 4 7 6 1 2 2 1 3 1 3 2 1 2 1 7 0 9 7 4 1 9 8 1 3 7 9 3 2 2 1 2 3 3 2 4 3 2 5 5 4 1 3 5 4 3 4 1 5 0 11 1 5 8 1 1 7 3 7 6 8 3 2 1 3 1 3 1 9 0 9 6 9 3 1 6 1 6 6 4 1 3 2 1 1 3 1 2 3 1 5 0 7 1 5 8 5 5 9 7 1 1 3 1 3 1 2 1 1 3 4 1 9 0 9 9 1 5 3 3 1 5 2 6 3 2 1 2 2 3 3 2 1 1 9 0 8 6 6 4 1 3 3 4 6 1 3 1 2 1 3 2 1 2 1 5 0 8 6 2 3 5 6 1 6 5 2 3 1 1 3 1 4 3 4 3 5 1 9 0 8 1 6 1 9 7 5 1 5 1 1 2 1 1 1 1 1 1 1 6 0 7 1 3 8 6 3 8 7 1 1 2 2 1 1 1 5 0 6 7 1 6 6 2 6 2 2 1 2 1 4 2 4 4 3 3 6 1 9 0 8 4 3 1 3 2 3 9 4 2 2 3 1 2 1 1 1 3 1 9 0 6 4 9 5 8 2 1 2 1 3 1 3 3 2 1 3 1 9 0 9 9 7 3 8 1 1 2 8 9 3 1 2 3 3 3 3 3 2 3 3 2 2 3 3 3 7 1 8 0 7 7 4 7 9 5 1 8 1 1 2 3 2 1 1 2 1 9 0 10 1 6 3 2 6 1 7 5 6 7 1 2 1 2 1 1 1 3 2 1 8 0 11 2 3 7 1 5 4 8 7 2 7 1 1 3 3 1 2 3 1 1 1 3 4 2 5 2 1 5 5 4 1 6 4 6 3 5 3 3 7 1 7 0 11 2 7 1 4 3 9 8 3 7 9 8 1 1 3 2 2 1 3 1 7 0 8 7 4 1 8 4 7 1 9 3 3 2 1 3 1 2 1 6 0 7 1 3 9 5 1 2 1 1 1 1 2 3 2 5 1 5 1 4 2 5 3 6 1 7 0 11 3 2 9 1 5 7 2 7 7 4 2 1 1 3 2 1 2 3 1 5 0 7 2 1 7 8 1 7 9 3 1 2 1 2 1 9 0 9 7 8 1 1 2 7 9 1 4 3 3 2 2 1 3 1 2 1 5 2 1 3 3 1 3 5 1 6 0 7 8 4 4 1 8 1 7 2 1 1 2 3 3 1 8 0 8 3 9 9 1 8 2 7 9 1 1 3 3 1 2 3 3 1 9 0 10 6 9 7 1 4 7 9 7 6 3 2 3 1 1 1 1 1 3 3 5 4 5 1 4 3 4 1 8 0 7 4 1 5 6 1 2 8 1 3 3 3 3 3 1 1 1 6 0 6 5 1 5 1 5 3 1 1 1 2 1 2 1 9 0 11 9 1 1 5 1 9 9 5 6 8 7 1 3 2 1 1 2 1 2 1 2 4 3 3 3 4 1 9 0 6 4 5 2 2 1 9 1 1 1 1 2 2 3 2 2 1 7 0 9 1 5 2 5 4 1 4 4 6 2 2 2 2 1 1 3 1 7 0 10 1 1 4 3 9 7 2 6 6 8 1 1 2 3 3 1 1 2 1 1 1 4 3 2 5 3 3 5 1 5 0 11 4 4 4 2 2 4 7 8 9 1 2 1 3 1 1 1 1 7 0 6 7 7 8 6 8 1 2 2 3 2 1 1 1 1 8 0 10 8 1 4 2 5 7 4 4 5 2 3 3 1 1 2 2 3 2 1 1 4 4 3 3 5 1 7 0 10 7 6 9 2 4 1 6 4 6 5 2 3 1 3 1 1 1 1 8 0 8 5 3 7 6 4 3 9 1 2 1 1 3 1 2 1 1 1 5 0 8 1 4 2 4 1 2 6 2 2 1 1 1 2 5 5 1 3 3 3 6 1 8 0 11 4 4 5 1 3 6 6 8 7 7 1 1 2 2 1 1 1 3 2 1 9 0 11 5 4 6 2 6 1 5 8 4 4 7 3 1 1 3 3 2 2 2 3 1 9 0 10 2 9 2 5 9 1 1 5 3 6 3 1 3 3 1 3 2 1 3 1 5 4 3 5 2 3 5 1 9 0 10 1 7 2 2 9 8 1 7 1 7 1 3 3 3 3 2 1 2 3 1 7 0 6 2 6 1 4 3 2 1 1 1 3 1 2 1 1 9 0 9 2 7 5 4 9 1 2 5 8 1 2 3 2 1 1 2 2 1 5 4 3 1 4 3 6 1 6 0 11 9 6 9 1 7 8 8 1 7 9 4 1 1 2 1 1 2 1 8 0 8 2 5 9 6 2 8 1 9 3 1 2 1 3 1 2 2 1 5 0 11 5 1 6 8 6 4 5 6 4 7 1 3 1 3 1 1 5 1 4 4 3 3 6 5 4 5 4 3 6 1 8 0 9 8 5 4 5 4 7 7 1 5 2 2 1 1 1 3 3 1 1 9 0 6 1 2 9 9 4 1 2 2 2 2 1 1 1 3 3 1 6 0 6 1 1 2 3 1 5 3 3 3 1 3 2 4 2 3 3 4 3 3 4 1 9 0 6 1 3 3 3 5 7 1 1 1 2 1 1 1 1 1 1 7 0 7 3 1 1 9 6 2 3 3 1 3 1 2 2 3 1 7 0 10 2 8 1 8 8 9 5 5 4 6 2 3 3 1 2 3 2 3 3 5 3 3 4 1 5 0 11 1 4 1 6 3 5 5 2 1 7 7 1 1 1 2 2 1 8 0 8 4 5 4 3 6 2 1 3 3 2 2 3 1 1 1 1 1 8 0 9 4 5 8 1 5 7 2 4 9 1 2 3 2 1 3 3 2 2 2 2 1 3 7 1 9 0 7 1 8 5 1 2 4 7 2 1 3 3 1 3 3 1 1 1 8 0 7 6 7 2 1 9 1 1 3 3 2 2 1 3 1 3 1 5 0 6 9 3 7 6 3 1 2 3 2 3 1 4 4 2 3 1 3 4 3 4 1 8 0 10 5 3 1 2 1 3 8 8 6 1 3 1 3 2 3 2 1 2 1 8 0 8 8 2 1 3 7 2 7 2 1 1 3 3 1 1 1 1 1 9 0 10 5 1 6 8 6 8 3 4 1 5 2 2 1 2 1 1 1 1 1 5 3 5 2 1 3 6 2 4 3 3 4 1 5 0 8 1 6 2 1 6 1 3 3 2 1 3 1 2 1 8 0 7 5 7 1 9 2 4 6 3 3 3 3 2 1 1 1 1 6 0 6 7 4 7 1 5 7 1 3 1 2 1 1 3 5 3 3 3 4 1 7 0 7 1 4 1 7 5 4 6 1 2 3 2 1 3 3 1 5 0 10 6 9 4 3 3 7 4 2 9 1 1 1 1 3 2 1 9 0 11 5 8 2 1 2 8 5 3 7 6 7 1 1 1 1 2 2 2 3 1 5 3 4 2 3 5 1 8 0 8 6 2 9 8 7 8 3 1 2 2 2 3 3 2 1 3 1 7 0 10 2 5 6 5 9 4 1 4 3 2 2 1 1 3 1 1 2 1 5 0 11 9 8 3 1 9 9 3 8 9 3 4 1 1 3 1 2 2 5 2 5 2 3 4 1 7 0 10 1 6 5 4 3 6 4 3 4 4 3 2 3 1 1 2 2 1 9 0 6 4 4 7 8 1 1 2 2 2 1 1 2 1 3 2 1 7 0 6 8 1 3 5 7 8 2 3 3 3 3 1 3 2 3 2 3 1 4 5 5 3 3 5 1 5 0 8 8 6 1 8 1 7 7 1 1 2 1 1 1 1 6 0 11 3 7 4 6 1 2 8 4 9 7 4 1 1 3 1 1 3 1 7 0 9 5 9 2 9 9 2 3 3 1 3 2 1 2 1 2 3 2 5 2 5 4 3 7 1 5 0 10 3 3 7 6 5 2 2 1 9 8 3 3 2 3 1 1 5 0 9 3 5 2 9 4 3 1 5 3 2 2 1 3 2 1 5 0 6 8 9 4 1 4 6 1 1 3 2 2 1 4 3 2 2 4 1 3 7 1 7 0 9 9 3 1 1 4 9 7 5 6 3 2 1 1 3 1 1 1 7 0 6 1 8 9 2 3 9 1 2 2 1 1 3 2 1 9 0 8 9 7 2 1 4 3 8 7 2 1 2 1 2 3 1 2 1 3 5 2 2 1 4 3 3 7 1 8 0 8 4 4 1 4 7 9 8 9 1 2 2 1 3 1 2 2 1 9 0 8 1 1 1 7 9 1 3 2 2 3 1 1 1 3 3 3 3 1 8 0 11 5 8 9 7 9 4 7 2 2 1 8 3 1 2 2 3 2 3 1 1 3 5 5 5 1 1 3 7 1 8 0 9 6 7 1 4 1 4 7 1 9 1 3 3 3 1 2 1 1 1 8 0 11 3 1 2 6 8 4 8 3 7 1 4 2 1 2 1 2 2 3 2 1 5 0 6 2 5 4 9 1 1 1 1 1 1 2 1 2 2 2 2 4 2 5 7 3 4 5 3 6 1 7 0 9 6 6 7 4 1 6 6 4 1 3 3 3 2 1 1 3 1 7 0 10 3 4 1 5 1 9 3 3 4 7 2 3 2 3 1 2 1 1 5 0 10 3 7 9 1 7 2 3 6 5 2 2 1 3 3 3 5 2 2 2 3 2 3 6 1 9 0 9 6 9 4 2 9 3 4 3 1 3 2 3 3 3 3 1 3 1 1 6 0 11 1 9 8 9 9 2 8 9 1 1 2 3 2 1 1 2 3 1 5 0 10 9 5 4 4 1 1 7 6 4 5 3 2 2 1 1 3 1 4 4 2 3 3 7 1 5 0 6 2 1 3 2 7 4 2 2 3 1 2 1 6 0 7 1 2 4 2 9 7 3 2 1 2 1 1 1 1 8 0 7 6 3 1 8 8 1 4 1 2 1 3 1 2 3 2 4 4 3 2 1 3 1 3 7 1 6 0 10 2 6 1 5 9 7 8 8 6 3 3 3 2 1 2 1 1 6 0 8 8 1 5 4 5 1 3 7 1 3 3 1 2 2 1 5 0 9 3 6 1 5 2 3 1 6 8 1 3 2 1 2 2 2 1 5 4 2 2 3 2 6 2 2 1 5 8 6 3 5 3 3 5 1 5 0 9 8 9 3 9 2 9 4 9 1 3 2 1 1 1 1 8 0 8 1 9 4 8 7 1 8 4 2 3 3 3 2 2 1 1 1 6 0 9 7 7 4 1 5 2 6 3 8 3 3 1 1 2 2 1 2 3 1 4 3 7 1 8 0 7 2 1 5 9 7 4 1 3 1 1 3 2 1 2 1 1 7 0 7 8 3 4 1 1 9 6 2 1 1 2 2 3 3 1 5 0 7 5 4 2 7 3 1 8 2 1 2 1 1 3 3 3 1 4 1 4 3 4 1 5 0 11 8 3 6 4 6 5 8 1 3 6 5 1 3 1 1 3 1 5 0 9 5 6 4 2 5 3 1 1 6 1 3 2 2 2 1 8 0 6 9 7 7 5 1 8 1 1 1 2 3 1 2 2 5 1 3 3 3 4 1 5 0 10 7 6 8 6 6 6 1 5 8 6 1 3 3 1 2 1 7 0 8 4 6 5 8 3 1 9 2 2 3 3 1 3 2 2 1 9 0 7 6 5 2 9 2 1 6 1 1 3 3 2 1 2 1 1 2 5 1 3 3 4 1 9 0 10 6 1 6 5 5 8 1 7 5 3 1 2 3 1 1 2 1 2 1 1 6 0 8 3 1 8 5 7 1 2 5 1 2 2 2 3 3 1 9 0 7 5 6 4 9 1 1 3 2 1 2 1 3 3 1 3 3 4 2 5 3 5 1 3 5 4 3 7 1 7 0 7 2 5 3 1 7 3 3 1 2 1 1 2 2 2 1 9 0 7 1 7 1 9 2 2 8 3 3 3 2 3 1 3 1 2 1 8 0 11 7 1 8 5 7 8 6 4 8 9 2 3 3 1 1 1 1 1 2 3 2 1 3 3 3 3 3 5 1 8 0 6 1 5 6 8 5 3 3 1 1 1 1 2 1 3 1 7 0 6 1 4 8 4 9 7 1 2 2 3 1 2 1 1 7 0 10 9 1 3 9 5 6 7 7 8 1 3 3 1 2 1 1 2 4 2 2 1 4 3 5 1 7 0 10 7 8 5 9 6 3 3 1 9 7 2 2 2 1 2 1 1 1 9 0 11 8 5 5 5 9 4 4 1 3 8 6 1 1 1 1 2 3 2 2 2 1 9 0 11 1 9 7 4 5 6 1 4 6 8 6 1 2 2 3 2 2 1 2 1 1 3 3 1 2 3 5 1 9 0 10 1 2 4 4 5 5 5 3 5 1 2 1 3 3 1 1 2 1 2 1 5 0 7 9 3 3 1 6 3 6 3 1 3 1 1 1 9 0 9 9 8 1 1 6 9 1 7 1 2 2 1 3 1 2 3 2 2 2 5 2 5 1 3 7 1 8 0 8 7 3 2 5 4 8 9 1 3 2 2 1 1 3 1 3 1 8 0 7 9 9 9 4 2 5 1 1 2 3 2 2 1 2 1 1 5 0 10 7 5 9 7 9 4 2 4 1 1 1 2 2 1 3 5 1 4 3 2 2 4 2 7 4 1 5 5 3 4 1 7 0 9 1 8 3 1 1 5 6 6 7 1 1 3 3 2 2 2 1 6 0 8 3 1 8 1 3 1 1 1 1 1 2 1 2 1 1 6 0 10 8 1 5 7 3 7 7 7 8 6 1 1 1 2 3 1 2 1 5 3 3 5 1 9 0 6 7 4 9 6 1 9 1 3 1 1 1 3 1 2 1 1 8 0 10 1 3 1 8 1 2 9 2 6 1 2 1 3 1 2 3 3 1 1 8 0 6 6 6 9 4 1 2 1 3 3 1 1 2 1 1 5 3 3 2 3 3 5 1 8 0 11 7 1 8 3 9 2 9 7 7 9 6 3 2 1 1 3 1 2 3 1 7 0 7 6 7 9 7 6 1 1 1 1 2 3 1 3 3 1 5 0 10 7 7 1 1 7 4 6 3 3 3 1 1 1 3 2 3 4 4 3 3 3 4 1 5 0 7 1 9 2 5 8 2 1 2 1 1 2 3 1 8 0 7 9 5 1 1 6 8 4 1 2 3 2 1 1 2 3 1 5 0 6 5 9 1 5 2 1 3 1 1 2 3 5 2 1 3 3 4 1 7 0 9 1 1 2 8 8 2 1 4 8 2 1 3 1 2 1 1 1 8 0 10 5 8 9 4 1 4 8 6 3 4 3 2 2 1 3 2 3 2 1 8 0 7 4 1 9 4 5 5 3 2 1 1 1 3 2 3 1 2 1 2 1 3 1 6 5 3 5 5 3 5 1 8 0 6 6 2 3 1 9 8 1 1 3 2 1 3 2 2 1 7 0 10 1 6 7 9 1 2 1 1 2 5 2 1 1 2 3 1 3 1 5 0 7 4 6 8 1 4 9 7 2 3 2 2 1 5 3 1 3 2 3 5 1 5 0 9 3 1 5 1 9 3 7 7 3 2 1 1 2 3 1 7 0 7 9 4 2 9 4 1 8 3 1 1 1 3 3 1 1 9 0 6 1 1 7 2 3 4 2 3 3 2 2 1 1 1 1 1 4 5 2 3 3 6 1 7 0 7 1 4 1 4 9 4 1 1 1 3 2 1 1 2 1 9 0 6 8 4 1 3 4 6 1 2 2 1 1 1 3 2 3 1 5 0 8 9 7 5 4 2 3 1 6 3 3 3 1 1 2 3 5 5 3 2 3 6 1 9 0 7 8 3 1 1 7 6 9 3 3 1 2 2 2 1 1 1 1 9 0 10 1 9 4 6 2 7 6 8 4 9 3 3 1 2 1 3 3 1 1 1 5 0 11 4 2 5 3 1 5 5 9 9 7 5 1 1 2 3 3 5 2 3 5 2 4 3 6 1 6 0 9 4 6 1 7 1 3 1 2 5 2 1 1 3 3 2 1 9 0 6 1 6 9 1 1 1 2 3 2 1 2 1 1 2 2 1 6 0 6 8 8 8 3 6 1 1 3 1 2 2 1 2 5 4 2 5 5 7 4 5 3 3 5 5 3 6 1 5 0 9 1 1 2 3 3 6 3 2 6 3 2 1 2 1 1 6 0 11 9 1 4 9 2 4 1 1 5 5 9 1 3 3 2 3 2 1 7 0 10 5 6 9 5 2 4 1 1 6 5 2 1 2 3 3 3 1 1 1 3 1 1 3 3 6 1 5 0 9 7 8 5 9 7 4 8 2 1 2 1 1 1 3 1 8 0 8 9 4 3 4 6 1 8 3 1 1 1 1 2 1 2 1 1 8 0 11 4 9 2 5 4 9 6 1 6 7 2 2 3 2 2 1 1 1 1 1 4 3 3 1 4 3 7 1 8 0 11 6 6 4 5 1 3 7 4 5 7 3 1 2 2 1 2 1 3 3 1 8 0 8 9 4 4 6 1 9 3 7 1 2 3 3 3 3 3 1 1 7 0 10 2 6 9 3 1 6 8 4 4 3 1 2 1 1 2 2 2 1 1 2 1 4 1 3 3 4 1 6 0 10 4 2 9 8 8 1 1 2 2 4 3 2 3 2 1 1 1 8 0 10 4 3 2 6 7 8 1 6 2 7 1 2 1 2 3 1 3 3 1 7 0 7 4 2 6 4 1 4 6 1 2 1 2 2 2 3 5 4 3 2 3 5 1 8 0 11 8 2 8 2 4 1 4 6 9 1 2 3 1 2 3 1 2 2 2 1 8 0 9 5 3 9 5 7 1 1 7 5 3 3 1 3 1 1 3 1 1 6 0 11 3 9 9 3 7 1 5 2 1 5 6 3 2 2 1 1 3 5 1 1 1 2 1 4 7 3 4 4 3 3 5 1 9 0 11 2 6 3 8 1 5 8 8 8 6 4 3 3 1 1 1 2 1 2 1 1 8 0 6 1 8 2 9 1 8 1 2 2 1 1 2 3 2 1 7 0 6 7 6 9 1 3 6 1 2 1 2 1 1 2 3 2 4 2 2 3 6 1 9 0 9 1 9 3 8 6 4 8 2 3 1 2 1 3 2 1 2 1 3 1 6 0 8 2 1 5 2 5 1 3 5 1 2 2 2 3 2 1 6 0 8 1 1 5 3 6 1 6 5 3 1 2 1 2 2 1 4 1 4 3 5 3 6 1 5 0 9 8 8 1 6 3 6 1 6 2 1 1 2 1 2 1 9 0 10 3 5 4 1 6 3 9 1 8 9 3 1 3 2 1 1 2 2 1 1 6 0 6 1 4 3 5 2 1 1 1 1 1 1 1 1 1 5 3 2 1 3 5 1 8 0 8 3 7 1 6 4 3 4 1 2 1 1 1 3 3 1 3 1 7 0 10 4 5 6 2 5 1 8 5 1 7 3 2 3 1 2 3 1 1 7 0 6 1 4 2 1 5 1 2 1 1 3 1 1 1 3 3 2 1 2 6 6 4 3 2 4 6 3 5 4 3 4 1 8 0 6 1 1 1 8 3 5 3 1 1 3 2 1 2 1 1 7 0 6 8 8 2 5 1 5 3 3 1 1 3 1 3 1 8 0 11 3 6 9 6 6 5 3 1 5 5 9 1 1 2 2 2 3 1 3 2 1 3 5 3 4 1 9 0 10 2 1 2 3 2 9 9 3 6 1 1 2 1 2 3 2 2 2 2 1 7 0 11 7 6 8 1 8 1 7 4 9 4 7 2 1 3 3 1 3 1 1 6 0 7 4 9 1 5 6 1 1 3 3 2 1 3 3 3 2 1 1 3 4 1 8 0 6 3 1 7 3 2 1 3 1 1 2 3 1 1 1 1 9 0 9 9 6 1 3 6 8 7 5 8 2 3 1 1 2 2 2 3 1 1 5 0 11 4 2 5 9 1 5 1 1 7 4 9 1 1 1 1 2 5 2 5 1 3 6 1 7 0 8 2 6 2 6 5 1 9 5 1 3 1 1 1 1 3 1 9 0 9 6 9 1 1 1 8 1 5 8 2 2 1 2 1 1 2 1 3 1 9 0 8 5 9 5 1 1 8 2 5 3 3 3 1 1 3 1 3 3 1 1 5 3 2 5 3 7 1 7 0 7 7 2 8 1 3 2 3 3 2 2 3 1 2 1 1 5 0 7 3 4 1 3 3 1 3 2 3 2 1 1 1 8 0 11 1 6 7 5 6 6 1 5 5 5 8 1 2 3 2 2 1 3 1 3 5 2 1 1 1 1 7 3 7 4 4 3 3 7 1 6 0 11 3 2 9 6 7 7 7 9 9 3 1 2 3 1 3 1 1 1 5 0 9 4 1 4 9 3 8 5 2 4 1 1 3 3 3 1 7 0 8 4 1 3 1 4 5 3 6 3 1 2 2 1 3 3 5 1 2 2 1 5 3 3 7 1 7 0 9 8 1 5 5 8 4 2 5 1 1 1 3 3 2 3 2 1 6 0 11 9 5 4 4 8 3 7 9 1 4 1 2 1 2 2 1 3 1 5 0 8 1 7 7 6 9 2 8 5 1 1 2 2 1 3 1 1 1 1 1 5 3 7 1 7 0 8 4 3 4 6 6 1 4 4 3 2 1 1 2 1 3 1 5 0 6 4 4 1 7 5 1 2 1 1 3 2 1 8 0 9 2 3 7 7 8 8 1 4 8 1 1 1 3 3 2 2 1 3 4 4 3 3 5 2 3 7 1 6 0 7 6 1 6 1 3 8 1 1 2 1 3 1 1 1 5 0 9 2 6 7 1 9 2 4 9 7 1 1 3 1 2 1 9 0 9 2 9 1 2 7 9 1 8 8 1 1 1 2 2 1 1 1 3 3 3 3 2 5 5 1 4 3 3 5 5 3 4 1 5 0 11 6 1 7 6 6 4 4 4 5 8 9 1 1 2 3 1 1 5 0 8 6 4 7 6 1 2 8 1 2 1 1 2 1 1 8 0 9 1 7 2 8 4 6 3 5 1 3 1 2 1 1 3 3 3 1 2 4 1 3 4 1 8 0 10 6 8 8 9 2 1 9 2 6 8 1 2 1 1 3 1 2 2 1 6 0 7 8 1 1 4 5 4 9 1 2 1 3 1 3 1 6 0 11 5 7 4 2 1 9 1 5 5 7 2 2 1 3 2 3 3 5 2 1 2 3 4 1 6 0 11 8 4 4 1 7 9 2 9 4 5 9 1 3 2 1 3 1 1 5 0 11 5 4 9 1 6 1 1 5 7 8 5 1 3 3 1 1 1 9 0 6 5 7 1 2 5 1 1 1 1 3 3 2 1 2 3 1 2 4 3 3 5 1 5 0 8 6 1 3 4 7 1 3 9 1 1 3 2 3 1 6 0 6 1 6 9 8 8 9 1 2 1 2 1 1 1 7 0 6 3 1 8 4 3 1 3 1 3 1 1 3 3 1 5 3 2 1 3 7 1 9 0 9 3 7 9 3 7 7 1 8 5 2 2 1 3 1 1 3 1 2 1 6 0 10 1 6 3 6 1 6 4 1 3 2 3 2 1 3 1 3 1 6 0 8 8 7 6 3 2 5 7 1 2 2 1 1 1 2 5 1 3 3 5 3 2 1 1 4 2 3 5 4 3 7 1 7 0 11 1 7 8 4 9 1 4 6 3 6 5 3 3 1 2 3 1 1 1 8 0 6 4 3 4 5 8 1 1 1 2 3 3 1 1 1 1 6 0 7 1 1 6 1 4 6 8 3 3 1 1 2 2 2 1 2 4 1 1 3 3 5 1 7 0 11 1 5 2 1 4 1 3 2 3 9 3 2 2 3 2 1 3 3 1 9 0 7 5 1 5 4 7 5 2 2 2 2 1 1 1 2 1 1 1 6 0 10 6 1 5 7 6 7 8 4 6 5 1 2 1 3 2 1 3 2 4 1 2 3 6 1 5 0 11 8 1 1 4 8 2 9 8 5 8 9 2 1 2 1 3 1 5 0 9 8 8 5 1 3 6 8 1 6 2 3 1 1 2 1 9 0 11 1 9 3 6 2 5 3 8 4 6 6 2 1 1 3 2 3 2 1 2 1 2 2 4 1 3 3 4 1 8 0 9 3 5 9 1 6 5 8 3 3 2 3 1 2 3 2 2 1 1 8 0 8 9 5 6 8 8 1 4 8 3 1 2 3 3 3 1 1 1 8 0 9 5 4 7 8 2 6 5 7 1 3 3 3 2 1 2 2 1 1 5 3 4 3 4 1 9 0 10 6 4 7 1 2 1 4 8 8 4 3 1 1 3 1 2 1 3 1 1 7 0 8 2 7 7 1 6 6 1 8 1 1 1 3 2 3 2 1 7 0 11 2 2 1 4 6 6 1 4 2 1 4 2 3 1 2 1 2 1 2 1 1 4 2 3 3 7 5 4 3 6 1 7 0 6 7 4 7 4 6 1 1 1 2 3 2 2 1 1 5 0 11 1 5 5 9 2 2 8 1 5 1 1 1 3 2 1 3 1 7 0 9 6 3 5 1 3 1 7 2 5 2 2 2 3 1 2 1 1 1 3 1 1 1 3 4 1 5 0 6 5 5 5 1 5 5 1 3 1 2 1 1 5 0 9 2 9 4 5 8 6 1 4 8 3 1 3 1 3 1 7 0 11 9 7 2 8 8 2 4 4 1 2 1 3 2 3 2 1 2 1 3 4 1 1 3 5 1 7 0 6 5 9 4 1 9 9 2 1 3 1 1 3 3 1 7 0 11 3 5 8 3 2 1 5 8 6 3 1 3 3 1 1 3 1 3 1 5 0 7 9 3 1 8 9 5 7 1 3 2 1 2 4 3 1 3 1 3 7 1 8 0 6 1 5 4 3 2 6 1 1 3 1 3 2 3 1 1 7 0 7 7 2 8 1 3 8 3 1 2 3 2 1 3 3 1 9 0 7 4 5 9 1 3 5 2 3 2 1 3 1 2 2 2 3 3 2 2 2 2 2 2 3 5 1 8 0 8 2 4 7 1 5 1 2 8 1 3 2 2 1 3 2 1 1 7 0 10 4 1 7 8 2 3 8 9 1 7 3 1 1 1 2 1 3 1 8 0 6 1 5 1 8 5 5 3 2 2 1 1 3 3 1 5 3 5 3 2 4 5 2 2 5 4 3 6 1 9 0 6 4 8 1 2 3 9 2 3 2 3 1 2 1 3 1 1 6 0 6 7 1 6 7 3 4 3 3 3 1 1 1 1 6 0 8 2 7 4 1 6 3 8 2 1 3 3 1 2 1 3 1 4 4 5 3 3 6 1 5 0 11 1 4 2 2 8 4 4 6 7 3 2 3 1 2 1 3 1 7 0 7 4 6 7 1 8 4 3 3 1 1 1 1 1 1 1 8 0 7 9 8 8 5 7 8 1 3 1 1 2 1 3 1 3 4 3 5 4 3 3 3 5 1 5 0 6 1 7 6 8 1 6 3 1 3 1 3 1 9 0 9 9 4 1 8 5 8 4 9 4 1 3 1 2 2 3 1 2 3 1 6 0 9 3 2 4 4 7 8 1 8 8 2 1 2 2 2 1 4 5 4 1 3 3 5 1 9 0 10 1 7 8 9 1 4 1 6 7 2 1 1 3 2 2 1 3 2 2 1 7 0 9 1 1 5 9 7 5 6 1 2 1 1 3 2 2 3 1 1 7 0 7 2 5 8 7 2 9 1 3 3 2 1 2 1 3 1 2 4 4 4 3 6 1 8 0 11 1 9 3 8 8 2 1 1 1 5 5 1 1 2 3 1 2 3 2 1 5 0 7 3 8 4 4 2 2 1 3 2 3 1 1 1 7 0 7 7 8 3 9 2 4 1 3 3 3 1 3 1 1 3 5 5 5 5 3 6 7 2 5 3 5 2 7 3 4 5 3 4 1 8 0 9 8 9 9 1 1 4 1 3 1 1 2 1 3 2 3 3 1 1 9 0 9 4 5 1 1 6 1 7 2 6 1 2 1 1 1 3 3 2 1 1 9 0 9 6 6 5 3 1 2 9 5 1 2 3 1 1 1 3 2 1 1 4 5 3 3 3 7 1 6 0 10 8 2 1 7 1 4 2 6 6 3 1 1 2 3 3 3 1 6 0 7 1 8 1 2 5 9 7 3 1 3 1 3 1 1 9 0 6 3 8 1 8 1 1 1 3 2 3 1 1 1 3 1 4 4 2 2 2 5 3 3 6 1 6 0 7 9 6 1 5 5 4 2 1 2 3 3 1 3 1 6 0 10 5 4 4 6 9 4 6 7 1 1 2 2 1 2 1 2 1 5 0 7 1 7 2 7 2 1 8 1 3 1 2 1 1 3 4 1 3 1 3 7 1 7 0 10 8 3 8 6 7 5 3 1 1 1 2 1 2 3 3 3 2 1 5 0 11 1 8 4 9 5 6 7 9 2 9 1 3 1 3 2 1 1 5 0 6 7 1 1 6 1 1 3 2 2 1 3 1 2 2 3 3 1 5 4 2 5 6 1 4 4 3 7 1 8 0 6 7 1 8 9 1 6 3 3 3 1 3 2 1 1 1 9 0 7 7 1 6 7 8 7 5 3 1 3 2 3 2 3 2 3 1 7 0 10 1 2 3 4 9 7 4 9 3 3 3 1 3 2 1 2 3 2 2 5 4 1 1 3 3 6 1 5 0 11 1 8 1 4 7 6 6 1 5 7 6 1 1 1 1 1 1 6 0 11 4 9 1 4 5 9 3 3 2 1 5 1 3 3 1 2 3 1 9 0 10 2 6 6 5 3 5 8 4 5 1 1 2 1 3 3 2 3 1 2 5 2 1 2 1 3 3 4 1 7 0 7 7 1 7 1 3 1 2 2 2 3 1 2 1 2 1 8 0 9 7 5 1 8 3 3 2 7 1 1 3 3 1 1 1 1 3 1 6 0 9 8 4 1 7 4 6 1 1 4 1 1 1 1 1 3 2 5 2 1 3 6 1 5 0 11 3 1 4 8 3 4 5 7 8 2 3 2 1 1 3 1 1 7 0 6 1 1 8 1 8 5 2 1 2 2 3 3 1 1 7 0 9 5 3 5 9 1 5 3 6 5 3 1 1 2 2 1 1 2 3 2 4 2 5 1 4 1 2 5 3 3 6 1 8 0 6 6 9 2 1 7 5 2 1 3 1 2 3 3 1 1 6 0 11 9 6 1 5 5 1 9 1 7 9 9 2 2 3 1 1 1 1 9 0 11 1 3 6 3 3 7 6 2 3 1 5 2 3 3 2 1 1 1 1 1 3 3 1 3 4 1 3 6 1 8 0 7 2 9 4 2 5 5 1 1 2 1 2 3 1 3 2 1 5 0 9 7 5 2 7 3 1 2 2 5 3 1 1 3 3 1 8 0 6 8 1 6 9 1 4 2 3 1 2 1 1 1 1 5 3 2 5 4 3 3 7 1 6 0 11 7 6 6 7 4 4 1 6 1 9 2 3 2 1 3 1 1 1 8 0 6 1 9 4 5 4 1 2 3 1 3 3 3 2 1 1 5 0 6 4 1 8 1 8 8 2 3 1 3 3 4 3 2 3 2 1 3 3 4 1 5 0 8 5 9 7 1 8 2 8 4 1 3 1 2 1 1 6 0 10 7 6 9 3 4 1 6 8 9 6 3 1 2 1 2 1 1 9 0 6 6 3 9 1 1 5 1 1 3 1 3 1 2 1 2 3 4 3 3 3 6 1 5 0 9 3 7 2 6 4 1 1 4 1 3 2 1 1 3 1 9 0 9 5 3 8 2 4 5 6 7 1 3 2 3 2 1 2 1 3 1 1 7 0 6 9 4 8 4 1 3 1 3 1 3 2 3 2 4 5 4 4 2 4 3 5 4 4 3 3 5 1 6 0 8 1 4 8 4 4 1 9 8 2 2 1 3 2 3 1 6 0 8 4 1 8 8 6 1 1 3 1 2 2 3 3 2 1 7 0 10 8 1 9 3 3 2 7 9 3 5 1 1 1 2 2 2 3 3 3 4 3 3 3 4 1 8 0 6 3 1 2 6 5 9 1 1 1 1 1 3 1 1 1 5 0 8 1 1 1 2 3 8 4 2 3 1 1 2 1 1 8 0 10 5 5 7 3 5 9 1 2 4 8 1 1 1 1 2 1 2 2 2 3 2 5 3 6 1 5 0 7 1 3 1 3 1 7 5 1 2 3 2 1 1 8 0 10 1 3 9 4 1 8 9 7 7 8 3 1 1 3 1 2 1 2 1 7 0 6 1 9 7 8 1 6 1 1 2 2 1 3 2 1 5 3 5 3 4 3 6 1 8 0 10 1 7 8 8 7 1 3 3 4 1 1 3 1 1 1 3 1 1 1 8 0 10 4 4 1 1 3 5 9 8 1 1 1 2 1 1 3 1 3 3 1 7 0 6 9 1 3 1 9 3 3 3 1 3 1 3 1 2 4 1 2 1 4 4 2 4 4 3 3 7 1 9 0 11 9 5 4 1 9 1 7 7 2 7 9 1 2 1 3 2 1 3 3 1 1 6 0 11 8 8 6 8 1 6 3 1 7 4 5 3 1 2 3 1 2 1 5 0 8 9 3 1 2 1 2 2 5 1 1 3 3 1 5 3 2 1 1 4 1 3 4 1 9 0 10 8 5 7 8 7 7 1 3 1 1 2 1 1 2 3 3 3 1 3 1 6 0 6 8 1 1 2 7 3 1 2 3 3 1 1 1 8 0 8 1 7 4 6 3 4 3 2 3 2 1 3 3 1 3 1 1 3 2 2 3 4 1 9 0 11 5 8 1 8 2 1 4 8 2 6 5 1 1 1 2 2 2 1 1 1 1 8 0 8 9 9 1 6 9 6 7 5 3 2 1 1 3 1 2 3 1 5 0 10 9 2 7 2 6 1 3 2 2 8 1 1 3 2 1 2 2 5 5 3 4 1 6 0 6 8 9 6 7 6 1 3 3 3 1 1 3 1 5 0 7 9 1 4 3 9 9 3 1 2 2 3 3 1 6 0 9 9 9 3 4 5 6 9 9 1 2 1 1 1 1 3 3 1 5 1 1 3 4 5 5 3 7 1 8 0 11 3 2 3 7 2 5 2 3 1 9 1 2 2 1 2 3 2 1 1 1 9 0 10 3 8 4 3 6 1 4 4 7 4 1 3 1 1 2 3 1 1 2 1 9 0 6 1 2 5 4 5 6 3 2 2 1 2 1 3 1 3 5 2 2 5 3 2 4 3 5 1 5 0 11 1 2 3 4 9 7 3 1 4 4 9 1 3 2 3 1 1 6 0 9 3 1 1 8 1 8 1 7 3 1 2 1 2 2 1 1 6 0 8 5 3 2 2 1 6 2 1 2 1 2 3 2 3 3 4 1 3 1 3 5 1 9 0 6 3 5 6 5 8 1 1 2 1 2 2 1 2 1 3 1 7 0 11 9 5 8 7 7 6 8 7 1 6 9 1 3 2 3 2 3 2 1 9 0 11 8 1 4 5 4 2 3 8 9 5 1 2 3 1 2 1 3 3 1 1 4 3 1 2 5 3 7 1 7 0 11 7 4 9 3 2 1 6 1 5 2 5 1 2 3 3 2 3 1 1 7 0 9 1 6 5 6 7 3 3 6 8 2 1 3 2 1 1 1 1 8 0 6 1 9 8 2 5 7 2 3 3 2 3 1 1 1 2 2 3 2 2 2 1 3 5 1 6 0 9 2 2 8 1 2 1 3 7 9 1 2 2 3 1 2 1 6 0 10 8 1 9 1 4 1 7 6 3 1 1 2 3 1 1 2 1 7 0 7 4 9 9 7 1 3 1 3 3 1 2 1 2 1 3 4 5 3 2 7 7 2 2 1 5 5 3 5 1 5 0 11 2 9 4 7 6 9 1 2 4 6 4 2 2 3 3 1 1 5 0 9 8 3 8 7 1 1 2 9 6 1 1 3 2 1 1 6 0 7 9 4 9 4 1 8 3 1 1 2 1 2 3 2 1 2 2 1 3 5 1 6 0 7 3 7 1 9 1 4 6 1 1 1 1 1 3 1 6 0 6 8 2 3 3 1 7 2 2 1 1 2 3 1 9 0 6 1 4 6 6 7 4 2 3 1 2 1 1 2 2 1 3 3 2 5 3 3 5 1 5 0 9 4 4 4 7 4 5 6 1 8 1 3 3 2 1 1 5 0 9 7 9 9 6 8 1 6 2 1 1 2 3 3 1 1 6 0 7 2 8 1 3 9 4 2 3 1 1 1 2 1 2 4 2 1 1 3 6 1 5 0 11 9 6 8 1 5 7 7 8 9 9 9 1 3 1 3 3 1 7 0 8 5 1 8 8 2 8 9 3 2 3 2 3 1 3 2 1 6 0 6 7 2 1 1 1 4 1 2 3 1 2 1 2 5 4 2 3 4 3 4 1 9 0 11 2 3 3 8 1 1 2 3 7 9 1 2 1 1 1 1 1 1 3 1 1 5 0 9 2 1 2 4 3 7 2 6 8 3 2 1 3 1 1 6 0 11 7 1 3 1 3 3 8 3 7 6 3 2 1 2 1 3 3 1 1 2 2 1 7 2 1 1 2 5 5 6 2 4 4 3 7 1 5 0 10 1 7 8 7 1 2 7 9 3 7 3 3 2 2 1 1 6 0 7 1 6 5 7 1 7 3 3 2 2 1 2 3 1 9 0 6 1 4 6 6 4 4 1 1 1 2 1 1 2 2 2 3 2 5 3 5 2 2 3 7 1 9 0 7 6 6 6 4 4 1 8 1 1 1 1 2 1 3 2 1 1 5 0 9 5 7 9 1 5 4 7 6 2 1 2 1 2 1 1 8 0 10 1 4 1 4 2 2 4 1 1 4 2 1 3 1 3 2 3 1 4 4 2 1 2 4 2 3 7 1 7 0 6 9 6 4 3 4 1 2 2 3 1 1 1 2 1 5 0 8 2 1 8 4 7 8 1 6 2 1 2 1 1 1 7 0 7 8 4 7 2 1 2 3 3 2 1 1 1 1 1 1 4 2 4 3 3 5 3 5 1 5 0 9 1 4 1 1 4 1 2 4 3 1 1 3 1 1 1 9 0 9 2 5 8 1 7 5 9 8 3 2 2 1 2 1 2 2 1 1 1 7 0 9 3 3 3 4 4 6 1 1 6 3 2 1 1 1 2 3 5 1 2 1 1 1 4 3 6 5 4 3 4 1 7 0 9 4 1 1 4 6 9 6 3 8 3 1 1 2 1 2 2 1 5 0 8 3 6 1 5 2 6 4 6 1 3 1 3 1 1 5 0 8 9 1 8 6 2 8 7 6 1 3 3 1 2 1 4 4 5 3 4 1 8 0 10 3 8 4 4 5 4 1 1 2 1 2 2 2 2 3 3 3 1 1 5 0 8 6 3 4 5 4 1 2 7 1 2 2 3 1 1 8 0 8 3 6 7 9 1 3 8 7 3 3 1 3 3 2 1 2 5 4 1 1 3 5 1 6 0 10 1 4 3 1 3 8 3 8 6 7 1 2 3 3 3 2 1 5 0 7 9 8 3 2 1 6 3 3 3 3 1 3 1 9 0 10 6 4 9 8 1 4 9 2 3 2 1 2 1 1 3 3 1 1 3 1 5 4 1 1 3 7 1 7 0 10 3 2 2 2 1 1 8 1 5 6 3 3 1 3 1 2 3 1 6 0 6 4 7 4 4 1 8 3 2 3 1 2 3 1 8 0 10 8 4 4 1 2 2 4 8 1 7 2 1 1 3 2 1 2 3 1 4 1 1 1 1 3 3 6 1 5 0 11 1 1 2 1 5 2 6 7 5 9 1 1 2 2 2 3 1 6 0 9 6 6 7 7 1 4 5 3 4 2 1 1 1 3 1 1 8 0 6 3 5 3 3 1 8 1 3 1 2 2 1 3 2 3 3 5 1 1 1 5 2 7 2 5 5 3 7 1 7 0 9 8 1 2 9 8 4 1 5 4 1 3 3 3 1 2 3 1 8 0 7 5 5 5 8 1 7 2 3 1 1 3 3 2 1 3 1 9 0 8 4 1 9 3 5 1 7 1 1 1 3 2 2 3 3 3 1 5 1 2 3 1 1 3 3 6 1 9 0 6 1 2 1 5 2 2 2 2 1 2 2 2 2 2 2 1 7 0 11 5 5 3 1 7 5 9 3 3 5 4 1 1 2 1 1 2 3 1 8 0 9 1 8 5 5 4 3 2 2 4 1 3 2 2 3 1 1 1 2 3 4 3 2 5 3 5 1 6 0 10 6 5 3 3 4 9 8 1 1 8 1 1 1 1 2 2 1 7 0 6 9 2 4 6 1 1 1 1 1 2 2 1 3 1 9 0 6 2 1 7 2 9 7 3 2 2 3 3 2 3 2 1 5 3 1 4 5 3 4 1 6 0 10 1 8 5 1 8 2 9 3 1 9 1 1 1 1 2 1 1 7 0 11 3 8 7 3 9 2 6 4 5 1 5 1 2 2 1 2 1 2 1 8 0 11 3 6 8 1 3 1 6 1 2 6 1 1 2 3 1 3 1 1 3 1 4 3 2 3 5 1 6 0 11 5 9 1 9 9 4 1 9 7 1 5 2 3 1 2 2 3 1 5 0 10 1 1 1 9 5 5 4 3 9 7 3 2 3 1 1 1 6 0 11 1 2 5 4 1 7 7 3 1 8 2 1 3 1 3 3 3 1 1 2 5 5 7 5 6 4 6 5 3 3 6 1 5 0 8 1 4 2 1 1 9 4 6 3 3 2 2 1 1 8 0 8 7 1 6 3 1 1 6 3 3 1 2 2 1 2 1 1 1 6 0 9 8 4 3 2 2 6 1 3 4 2 3 1 1 1 3 5 5 2 1 2 1 3 6 1 5 0 10 3 6 7 9 4 1 3 5 4 7 2 3 1 3 3 1 9 0 8 1 5 3 8 9 1 6 5 2 3 2 1 2 2 2 3 2 1 9 0 8 1 8 9 6 3 9 1 8 1 3 3 3 1 3 2 1 1 5 3 2 4 2 5 3 4 1 7 0 6 7 1 5 9 9 3 1 1 2 2 3 1 1 1 9 0 9 5 7 7 5 8 7 7 3 1 1 1 2 1 3 1 1 1 2 1 5 0 8 5 7 1 3 3 2 5 5 1 2 2 1 3 3 3 4 2 3 4 1 6 0 6 1 4 9 1 9 2 1 3 1 1 3 1 1 5 0 7 6 1 7 7 6 9 1 1 1 3 2 2 1 6 0 8 6 8 1 6 3 8 1 6 1 1 1 3 2 2 3 3 4 1 3 7 1 7 0 9 8 8 6 3 1 5 8 8 7 1 2 1 1 3 2 3 1 9 0 7 8 1 2 1 9 5 6 2 1 3 1 1 3 3 1 1 1 6 0 10 1 2 6 3 1 5 4 8 4 7 2 2 1 1 3 1 4 3 2 2 1 2 3 3 4 4 4 4 3 4 1 6 0 6 6 3 1 5 3 1 2 1 2 3 3 1 1 6 0 7 5 6 5 6 1 1 6 1 2 1 1 3 3 1 7 0 9 1 6 1 2 5 9 4 7 2 3 1 2 1 1 1 2 3 1 3 3 3 5 1 6 0 10 1 4 5 1 9 2 1 7 1 2 2 2 1 1 3 1 1 7 0 6 3 3 1 1 7 4 1 3 3 1 1 2 3 1 9 0 11 9 6 9 2 1 5 5 4 1 6 2 2 2 3 3 2 3 1 1 1 1 4 2 2 2 3 6 1 7 0 11 7 6 3 5 8 2 3 1 3 3 1 3 1 1 3 3 1 2 1 7 0 6 8 1 4 3 4 6 1 1 2 2 2 2 1 1 7 0 7 1 2 2 1 1 8 7 3 2 3 1 3 3 3 5 4 4 1 3 1 3 7 1 5 0 6 5 7 1 9 1 9 3 1 1 1 2 1 8 0 7 2 3 4 1 1 5 9 2 2 1 3 3 1 3 2 1 6 0 8 7 9 6 1 5 6 9 6 3 2 1 2 1 2 3 5 3 4 3 3 1 4 4 4 2 5 5 3 6 1 7 0 9 8 8 7 5 3 9 1 4 2 3 1 1 3 1 2 1 1 7 0 11 4 8 3 9 1 2 1 6 5 6 2 3 3 3 2 3 1 2 1 9 0 10 1 8 9 4 5 1 2 1 2 8 2 1 3 2 1 2 1 3 1 2 1 2 4 2 1 3 4 1 9 0 10 7 1 5 2 9 1 3 5 3 8 2 2 1 1 2 3 3 3 2 1 5 0 9 8 1 3 1 2 1 1 3 8 3 1 3 1 1 1 7 0 8 3 1 4 8 5 5 9 2 1 3 3 1 3 3 3 1 5 1 5 3 6 1 9 0 9 7 2 1 7 8 6 4 7 4 3 3 2 2 1 2 3 2 3 1 6 0 8 4 3 3 9 1 1 1 2 2 1 1 3 1 3 1 9 0 10 1 1 3 5 6 7 3 9 5 5 1 3 3 1 3 1 3 1 2 4 2 5 1 3 1 3 5 1 6 0 7 1 5 4 6 6 8 7 1 2 3 2 2 1 1 8 0 11 9 1 2 9 1 5 6 5 3 7 7 1 3 1 3 2 3 2 2 1 8 0 7 3 9 1 7 1 8 3 2 2 1 2 1 2 1 2 2 4 2 2 3 3 4 1 5 0 10 1 2 7 2 4 3 3 5 6 3 1 1 3 3 1 1 6 0 10 7 4 8 5 2 1 4 1 9 4 1 2 2 2 1 1 1 8 0 7 1 5 7 3 6 2 2 3 2 2 1 1 1 3 1 2 1 2 3 3 2 2 4 5 4 5 11 7 7 10 8 6 11 5 1 10 5"""
solution = problem_fifteen(message)