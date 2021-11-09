'''
Multi-class Poisson Disk Sampling (Wei, 2009)
as described in https://www.microsoft.com/en-us/research/wp-content/uploads/2009/01/paper.pdf

for a dense grid, i.e. mostly every pixel should be filled

Implemented by Cindy M. Nguyen
'''
import numpy as np
import json
from collections import defaultdict

np.random.seed(123)


def build_r_matrix(r_set, n=2):
    # r_set is the user specified per-class values
    # r(k,j) specified the minimum distance between samples from class k and j
    # n is the dimensionality of sample space
    num_classes = len(r_set.keys())
    r_matrix = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        r_matrix[i, i] = r_set[i]

    # sort classes into priority groups with decreasing r
    priority_groups = defaultdict(list)
    for key, val in sorted(r_set.items()):
        priority_groups[val].append(key)

    # make a sorted list of values
    p_list = []
    for item in sorted(priority_groups.items()):
        key, val = item
        p_list.append([key, val])
    p_list = sorted(p_list)
    done_classes = []
    density_of_done = 0.0
    for k in range(len(p_list)):
        (r, curr_classes) = p_list[k]
        done_classes += curr_classes
        for c in curr_classes:
            # all classes in current priority group should have identical r value
            density_of_done += (1.0 / (r ** n))
        for i in curr_classes:
            for j in done_classes:
                # for each class you're looking at, we iterate through all
                # covered classes to add to the matrix
                if i != j:
                    r_matrix[i, j] = r_matrix[j, i] = 1.0 / (density_of_done ** (1.0 / n))  # r is symmetric
    return r_matrix



def find_most_underfilled_class(sample_count):
    return min(sample_count, key=sample_count.get)


def distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def fill_rate(c, sample_count, num_target_samples):
    return sample_count[c] / num_target_samples[c]


def removable(conflicts, s, c, r_matrix):
    for past_pt, past_c in conflicts.items():
        if r_matrix[past_c, past_c] >= r_matrix[c, c] or fill_rate(past_c) < fill_rate(c):
            return False
    return True


def fill_grid(samples):
    grid = np.ones((width, height), dtype=np.float32) * 10
    # use 10 to identify which resulting pixel should be filled manually
    for pt, c in samples.items():
        grid[pt] = c
    return grid


def multiclass_poisson(r_set):
    '''

    @param grid: sampling domain
    @param r_set: user specified params for intra-class sample spacing
    @return:
    '''
    num_classes = len(r_set.keys())
    r_matrix = build_r_matrix(r_set, n=2)
    samples = {}  # [coord: class]
    sample_count = dict.fromkeys(range(num_classes), 0)  # [class num: num samples for that class]
    max_trials = 2000000
    total_num_samples = 302500

    current_num_samples = 0  # count the number of samples of all classes currently

    t = 0
    past_num = 0
    repeat_count = 0
    while t < max_trials and current_num_samples < total_num_samples:
        if t % 1000 == 0:
            if past_num == current_num_samples:
                repeat_count += 1
            else:
                repeat_count = 0
            past_num = current_num_samples
            print(f'Trials: {t}, num_samples: {current_num_samples}')
            if repeat_count == 10:
                # early stopping if after 10000 trials, nothing has changed
                break
        pt = (np.random.randint(0, width), np.random.randint(0, height))
        t += 1
        c = find_most_underfilled_class(sample_count)
        can_add = True

        conflicts = {}

        for key, val in samples.items():
            past_pt = key
            if distance(pt, past_pt) < r_matrix[c, val]:
                can_add = False
                conflicts[past_pt] = val
        if can_add:
            current_num_samples += 1
            sample_count[c] = sample_count[c] + 1
            samples[pt] = c
        else:  # impossible to add another sample to class c
            # try to remove the set of conflicting samples
            if removable(conflicts, pt, c, r_matrix):
                sample_count[c] = sample_count[c] + 1
                samples[pt] = c
                for past_pt, past_c in conflicts.items():
                    samples.pop(past_pt)
                    sample_count[past_c] = sample_count[past_c] - 1

    print(current_num_samples)
    print(sample_count)

    grid = fill_grid(samples)
    np.save(f'grid_{width}.npy', grid)

    file = open('output.txt', 'w')
    file.write(f'Sample Count: {current_num_samples}/{total_num_samples}, {json.dumps(sample_count)}')
    file.close()

if __name__ == '__main__':
    width = 550
    height = 550

    # r_set must start at 0 and increase by 1 with each class
    # use to set radius for each class, in this example, we use 2px for all classes
    r_set = dict.fromkeys(range(8), 2)

    # each target sample count as determined by Eq(1)
    num_target_samples = dict.fromkeys(range(8), 37813)
    multiclass_poisson(r_set)
