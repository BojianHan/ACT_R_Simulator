import random
import string
import numpy as np
import theano
import theano.tensor as TT
import copy
import math

NIL = -1
DEFAULT_DICT = {str(k): 1.0*k/10 for k in range(0, 11, 1)}

prod_str="""
(P initialize-addition
   =goal>
      ISA         add
      arg1        =num1
      arg2        =num2
      sum         nil
  ==>
   =goal>
      ISA         add
      sum         =num1
      count       0
   +retrieval>
      ISA        count-order
      first      =num1
)

(P terminate-addition
   =goal>
      ISA         add
      count       =num
      arg2        =num
      sum         =answer
  ==>
   =goal>
      ISA         add
      count       nil
)

(P increment-count
   =goal>
      ISA         add
      sum         =sum
      count       =count
   =retrieval>
      ISA         count-order
      first       =count
      second      =newcount
  ==>
   =goal>
      ISA         add
      count       =newcount
   +retrieval>
      ISA        count-order
      first      =sum
)

(P increment-sum
   =goal>
      ISA         add
      sum         =sum
      count       =count
    - arg2        =count
   =retrieval>
      ISA         count-order
      first       =sum
      second      =newsum
  ==>
   =goal>
      ISA         add
      sum         =newsum
   +retrieval>
      ISA        count-order
      first      =count
)
"""

dm_str = """
   (a ISA count-order first 0 second 1)
   (b ISA count-order first 1 second 2)
   (c ISA count-order first 2 second 3)
   (d ISA count-order first 3 second 4)
   (e ISA count-order first 4 second 5)
   (f ISA count-order first 5 second 6)
   (g ISA count-order first 6 second 7)
   (h ISA count-order first 7 second 8)
   (i ISA count-order first 8 second 9)
   (j ISA count-order first 9 second 10)
   (second-goal ISA add arg1 5 arg2 2)
"""

# This function parse a module from a production rule
# Returns the remaining lines and 
#         the module dictionary that maps module slot name to operator and value
def parseModule(lines):
    # Read in the module name by remove = or + or - from the first character
    # and remove > from the last character
    module_name, lines = lines[0][1:-1], lines[1:]
    module_dict = dict()
    # Read in the module until either the next module or end of a production rule
    while (not lines[0].endswith(">") and not lines[0] == ")"):
        # Read the whole slot
        slot_value, lines = lines[0].split(), lines[1:]
        # Normalize the slot_value triple by adding "" to ones without an operator
        if len(slot_value) == 2: slot_value = [""] + slot_value
        # Ignore ISA statement
        if (slot_value[1] == "ISA"): continue
        # Add it to the dictionary
        module_dict[module_name + "." + slot_value[1]] = (slot_value[0], slot_value[2])
    return lines, module_dict

# This function parse one side of the production rule
# Returns the remaining lines and 
#         the dictionary of that side of the production rule
def parseSide(lines):
    side_dict = dict()
    # Identify the middle point and end of the production rule 
    while (not lines[0] == "==>" and not lines[0] == ")"):
        lines, module_dict = parseModule(lines)
        side_dict.update(module_dict)
    return lines, side_dict

# Extract the module/chunk/slots from the dictionary
# Return a dictionary of global module slots to its possible values and
#        a set of local binding
def extractFromDict(lhs_dict, rhs_dict):
    global_slots = dict()
    local_values = set()
    for s_dict in [lhs_dict, rhs_dict]:
        for key in s_dict:
            # Create a new slot
            if key not in global_slots: global_slots[key] = set()
            value = s_dict[key][1]
            if (value.startswith("=")):
                # Update local variables
                local_values.add(value)
            else:
                # Update the possible values for the slot if it is not a variable
                global_slots[key].add(value)
    return global_slots, local_values

# Parse a production rule
# Return the remaining lines and
#        the global module slots dictionary and
#        the production rule dictionary
def parseProductionRule(lines):
    # Read until a valid production rule
    while (not lines[0].startswith("(P")):
        lines = lines[1:]
    # Get the name of the production rule
    prod_rule_name, lines = lines[0].split()[1], lines[1:]
    # Parse LHS, ==>, RHS and )
    lines, lhs_dict = parseSide(lines)
    lines = lines[1:]
    lines, rhs_dict = parseSide(lines)
    lines = lines[1:]
    # Get the set of global slot names and local binding values
    global_slots, local_values = extractFromDict(lhs_dict, rhs_dict)
    # Production rule dictionary
    rule_dict = {"NAME": prod_rule_name, "LHS": lhs_dict, "RHS": rhs_dict, "LOCAL": sorted(local_values)}
    return lines, global_slots, rule_dict

# Parse a string containing multiple production rule
# Return a global module slots list that maps module slot name to list of possible values
#        a list of production rule dictionary
def parseFile(prod_str):
    # Split the production rule string into lines and do other preprocessing
    lines = [' '.join(line.strip().upper().split()) for line in prod_str.strip().splitlines()]
    # Initiate global chunk slots dictionary and production rule list
    global_slots = dict()
    prod_list = []
    # Read until the end
    while len(lines) > 0:
        # Parse a production rule
        lines, update_slots, rule_dict = parseProductionRule(lines)
        # Update the production rule list
        prod_list.append(rule_dict)
        # Update the global_slots dictionary
        for key in update_slots:
            if (key not in global_slots): global_slots[key] = set()
            global_slots[key].update(update_slots[key])
    # Convert this to a sorted list of a tuple containing the name and list of possible values
    global_slots = sorted([(key, sorted(global_slots[key])) for key in global_slots])
    return global_slots, prod_list

def parseDM(dm_str):
    lines = [' '.join(line.strip().upper().split()) for line in dm_str.strip().splitlines()]
    while (not lines[0].startswith("(")):
        lines = lines[1:]
    chunk_list = []
    while (len(lines) > 0):
        chunk, lines = lines[0], lines[1:]
        chunk = chunk[1:-1].split()
        new_chunk = dict()
        chunk_name, chunk = chunk[0], chunk[1:]
        while (len(chunk) > 0):
            token, chunk = chunk[0], chunk[1:]
            if (token == "ISA"):
                chunk = chunk[1:]
            else:
                value, chunk = chunk[0], chunk[1:]
                new_chunk[token] = value
        chunk_list.append((chunk_name, new_chunk))
    return chunk_list


# Find if the module slot represents discrete values or continous values in terms of numbers
# Return True if the list contains non numbers exclusing NIL
# Return False otherwise
def isDiscreteValues(values):
    if sum([(not s.isdigit() and s != "NIL") for s in values]) > 0: return True
    else: return False

# Returns a global variable map that maps the name to a tuple
#         containing the index of that slot among all the slots and
#                    the possible values to real number mapping for that slot
def global_possible_values(global_slots):
    global_var_map = dict()
    for i in range(len(global_slots)):
        # Get list of possible values
        name, possible_values = global_slots[i]
        if not isDiscreteValues(possible_values):
            # Default real number list
            possible_values = DEFAULT_DICT
        else:
            # Get a list of mapping from value to real number
            l = len(possible_values)
            possible_values = {possible_values[k]: 1.0*k/max(1, l-1) for k in range(0, l, 1)}
        # Add in the NIL value
        possible_values.update({'NIL': NIL})
        global_var_map[name] = (i, possible_values)
    return global_var_map

# Print the production rule given a vector represent the production rule
def printProductionFireVector(vector, prod_list):
    for i in range(len(vector)):
        if vector[i] == max(vector):
            print prod_list[i]['NAME'],

# Print the production rule
def printProductionRuleVector(prod_rule, global_var_map):
    for key in global_var_map:
        idx = global_var_map[key][0]
        if (prod_rule[idx] < -0.5): 
            print "%20s: NIL" % (key)
        else:
            print "%20s: %8d" % (key, int(round(10 * prod_rule[idx])))

# Print the production rule
def printProductionRuleVectorComp(prod_rule, correct_rule, global_var_map):
    total_error = 0
    for key in global_var_map:
        idx = global_var_map[key][0]
        raw_value = prod_rule[idx]
        total_error += abs(correct_rule[idx]-raw_value)
        print "%20s: %8.1f (%8.1f) [Error: %.3f]" % (key, raw_value, correct_rule[idx], abs(correct_rule[idx]-raw_value))
    print "[Total Error: %.3f]" % total_error    

# Generate a single piece of data
# Return a vector containing representation of the LHS
#        a vector containing representation of comparison matrix
#        a vector containing representation of which production rule fired
#        a vector containing representation of the RHS
def generateData(global_var_map, prod_list, idx=None):
    # Get number of production rule and number of global module slots
    num_prod_rules = len(prod_list)
    num_global_slots = len(global_var_map)

    # Select the production rule to generate data from
    if idx == None: rule_index = random.randint(0, num_prod_rules-1)
    else: rule_index = idx % num_prod_rules
    
    # Production rule dictionary and dictionary mapping local variables to real number value
    rule_dict = prod_list[rule_index]
    local_variable_bind = dict()

    # Initialize the data for LHS and Fire Vector
    prod_lhs_input = [random.choice([NIL,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]) for k in range(num_global_slots)]
    prod_fire_vector = [(k == rule_index) * 1 for k in range(num_prod_rules)]

    # Generate data from the LHS
    for key in rule_dict['LHS']:
        # Get the operator and value in ACT-R rule
        operator, value = rule_dict['LHS'][key]
        # Get the list of possible real number values and shuffle it
        (idx, possible_reals) = global_var_map[key]
        shuffled_values = random.sample(possible_reals.values(), len(possible_reals))
        # Check if the value is already either present in possible reals or local binding
        if value in possible_reals: selected_real_num = possible_reals[value]
        elif value in local_variable_bind: selected_real_num = local_variable_bind[value] 
        else:
            # Select a possible value from the shuffled value and make sure it is not NIL
            selected_real_num = NIL
            while (selected_real_num == NIL):
                selected_real_num, shuffled_values = shuffled_values[0], shuffled_values[1:]
            # Add to binding if necessary
            if value.startswith("="): local_variable_bind[value] = selected_real_num
        # If negated (DOES NOT SUPPORT < > currently)
        if operator == "-":
            # Find a non NIL value that is not the selected one
            # Need to make sure there is enough possible values
            while (shuffled_values[0] == NIL or shuffled_values[0] == selected_real_num):
                shuffled_values = shuffled_values[1:]
            # Select that value
            selected_real_num = shuffled_values[0]
        # Put the real number at that position in the vector
        prod_lhs_input[idx] = selected_real_num

    # Set up the RHS output based on the LHS
    # Flush all buffer that is NOT the GOAL buffer
    prod_rhs_output = copy.copy(prod_lhs_input)
    for key in global_var_map:
        if (key.startswith("GOAL.")): continue
        prod_rhs_output[global_var_map[key][0]] = NIL

    for key in rule_dict['RHS']:
        # Get the list of possible values
        (idx, possible_reals) = global_var_map[key]
        value = rule_dict['RHS'][key][1]
        # The RHS value should either be variable binded or decided in the production rule
        if value in possible_reals: selected_real_num = possible_reals[value]
        elif value in local_variable_bind: selected_real_num = local_variable_bind[value]
        else: print "ERROR!" * 1000
        # Put the real number at that position in the vector
        prod_rhs_output[idx] = selected_real_num

    # Change the LHS input to a numpy vector
    prod_lhs_input = np.asarray(prod_lhs_input)
    # Compute the comparison matrix
    np_tmp = np.tile(prod_lhs_input, (len(prod_lhs_input), 1))
    prod_comp = np.reshape(np.clip(100 * (np_tmp - np.transpose(np_tmp)), 0, 1), (num_global_slots ** 2))
    # Change the fire vector to a numpy vector
    prod_fire_vector = np.asarray(prod_fire_vector)
    # Change the RHS output to a numpy vector
    prod_rhs_output = np.asarray(prod_rhs_output)
    return prod_lhs_input, prod_comp, prod_fire_vector, prod_rhs_output

def generateRetrievalData(global_var_map, chunk_list):
    num_global_slots = len(global_var_map)
    chunk_idx = random.randint(0, len(chunk_list)-1)
    chunk = chunk_list[chunk_idx]
    chunk_vector = [0] * len(chunk_list)
    chunk_vector[chunk_idx] = 1
    chunk_output = [random.choice([NIL,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]) for k in range(num_global_slots)]
    chunk_name, chunk_dict = chunk[0], chunk[1]
    for slot in chunk_dict:
        key = "RETRIEVAL." + slot
        if key in global_var_map:
            idx, possible_reals = global_var_map[key]
            chunk_output[idx] = possible_reals[chunk[1][slot]]

    chunk_input = copy.copy(chunk_output)
    removed_amt = 0
    for slot in chunk_dict:
        key = "RETRIEVAL." + slot
        if key in global_var_map:
            idx, possible_reals = global_var_map[key]
            if (removed_amt < len(chunk_dict) - 1 and random.random() > (1.0/len(chunk_dict))):
                chunk_input[idx] = NIL
                removed_amt += 1
    return np.asarray(chunk_input), np.asarray(chunk_vector), np.asarray(chunk_output)

number_repeats = 3

def encode(l, repeats):
    return np.tile(l, repeats)

def decode(l, repeats):
    c, = l.shape
    blank = np.zeros((c/repeats))
    for i in range(repeats):
        blank += l[i*c/repeats:(i+1)*c/repeats]
    return blank/repeats

def nil_encode(l, repeats):
    if (number_repeats == 1):
        return l
    n, = l.shape
    l = l.tolist()
    A = []
    B = []
    C = []
    for e in l:
        if e < -0.5:
            A.append(NIL)
            B.append(NIL)
            C.append(NIL)
        else:
            A.append(1)
            B.append(2*e-1)
            C.append(1-2*e)
    return np.asarray(A+B+C)

def nil_decode(l, repeats):
    if (number_repeats == 1):
        return l
    n2, = l.shape
    n = n2 / 3
    l = l.tolist()
    A = l[:n]
    B = l[n:2*n]
    C = l[2*n:]
    D = []
    for i in range(n):
        if (A[i] < -0.5):
            D.append(NIL)
        else:
            D.append((B[i]+1)/4-(C[i]-1)/4)
    return np.asarray(D)

def repeats_encode(l, repeats):
    n, = l.shape
    base = math.ceil(11**(1.0/repeats))
    l = l.tolist()
    newl = []
    for i in range(repeats):
        for j in range(n):
            if (l[j] < -0.5):
                newl.append(-1)
            else: 
                rmd = (l[j] * 10) % base
                l[j] -= (rmd / 10.0)
                l[j] = l[j] / base
                newl.append(rmd/(base - 1))
    return np.asarray(newl)

def repeats_decode(l, repeats):
    n2, = l.shape
    n = n2 / repeats
    base = math.ceil(11**(1.0/repeats))
    l = l.tolist()
    newl = [0] * n
    for i in range(n2):
        if (l[i] < -0.5):
            newl[i % n] = -1
        else:
            newl[i % n] += l[i] * (base-1) * (base ** (i / n)) / 10.0
    return np.asarray(newl)

# Parse the file and get the possible values for the global slots
global_slots, prod_list = parseFile(prod_str)
global_var_map = global_possible_values(global_slots)
chunk_list = parseDM(dm_str)


num_chunks, num_rules, num_slots = len(chunk_list), len(prod_list), number_repeats * len(global_slots)

# Guidelines
# x: Input
# y: Chunk Vector
# z: Output
# h: Hidden layer

# Theano Inputs/Targets
x = TT.matrix()
y_target = TT.matrix()
z_target = TT.matrix()
h0 = theano.shared(np.zeros(num_chunks))
y0 = theano.shared(np.zeros(num_chunks))
z0 = theano.shared(np.zeros(num_slots))
# Weight Matrix
W_xh = TT.matrix()
W_hh = TT.matrix()
W_hy = TT.matrix()
W_hz = TT.matrix()
W_zh = TT.matrix()
W_zz = TT.matrix()
b_h = TT.vector()
b_y = TT.vector()
b_z = TT.vector()

def stepDM(x_t, h_tm1, y_tm1, z_tm1, W_xh, W_hh, W_hy, W_hz, W_zh, W_zz, b_h, b_y, b_z):
    h_t = TT.dot(x_t, W_xh) + TT.dot(h_tm1, W_hh) + TT.dot(z_tm1, W_zh) + b_h
    y_t = TT.dot(h_t, W_hy) + b_y
    z_t = TT.dot(h_t, W_hz) + TT.dot(z_tm1, W_zz) + b_z
    return h_t, y_t, z_t

[h, y, z], _ = theano.scan(stepDM, sequences=[x], outputs_info=[h0, y0, z0], 
  non_sequences=[W_xh, W_hh, W_hy, W_hz, W_zh, W_zz, b_h, b_y, b_z])
error = ((z_target - z) ** 2).mean().mean() + 0 * ((y_target - y)**2).mean() 
[G_xh, G_hh, G_hy, G_hz, G_zh, G_zz, G_h, G_y, G_z] = TT.grad(error, 
  [W_xh, W_hh, W_hy, W_hz, W_zh, W_zz, b_h, b_y, b_z])

trainDM_fn = theano.function([x, y_target, z_target, W_xh, W_hh, W_hy, W_hz, W_zh, W_zz, b_h, b_y, b_z], 
    [G_xh, G_hh, G_hy, G_hz, G_zh, G_zz, G_h, G_y, G_z, error])
testDM_fn = theano.function([x, W_xh, W_hh, W_hy, W_hz, W_zh, W_zz, b_h, b_y, b_z], [y, z])

weightsDM = [
np.random.uniform(size=(num_slots, num_chunks), low=-.01, high=.01),    #W_xh
np.random.uniform(size=(num_chunks, num_chunks), low=-.01, high=.01),   #W_hh
np.random.uniform(size=(num_chunks, num_chunks), low=-.01, high=.01),   #W_hy
np.random.uniform(size=(num_chunks, num_slots), low=-.01, high=.01),    #W_hz
np.random.uniform(size=(num_slots, num_chunks), low=-.01, high=.01),    #W_zh
np.random.uniform(size=(num_slots, num_slots), low=-.01, high=.01),     #W_zz
np.random.uniform(size=(num_chunks), low=-.01, high=.01),               #b_h
np.random.uniform(size=(num_chunks), low=-.01, high=.01),               #b_y
np.random.uniform(size=(num_slots), low=-.01, high=.01),                #b_z
]
#                        W_xh  W_hh  W_hy  W_hz  W_zh  W_zz    b_h    b_y    b_z
learning_rate =         [  3,   3,   3,   3,   3,   3,    3,    3,    3]
learning_rate_updates = [.99, .99, .99, .99, .99, .99, 0.99, 0.99, 0.99]

batch_size = 10
momentum = 0.90
time_len = 2
epoch = 0

for i in range(epoch):
    # Set up the batch cumulative gradients
    cum_grads = [
        np.zeros((num_slots, num_chunks)),    #W_xh
        np.zeros((num_chunks, num_chunks)),   #W_hh
        np.zeros((num_chunks, num_chunks)),   #W_hy
        np.zeros((num_chunks, num_slots)),    #W_hz
        np.zeros((num_slots, num_chunks)),    #W_zh
        np.zeros((num_slots, num_slots)),     #W_zz
        np.zeros((num_chunks)),               #b_h
        np.zeros((num_chunks)),               #b_y
        np.zeros((num_slots)),                #b_z
    ]
    updates = copy.deepcopy(cum_grads)
    cum_error = 0
    # Run a batch
    for j in range(batch_size):
        x_t, y_t, z_t = generateRetrievalData(global_var_map, chunk_list)
        x_t = nil_encode(x_t, number_repeats)
        z_t = nil_encode(z_t, number_repeats)
        tmp = trainDM_fn(np.tile(x_t, (time_len, 1)),
                         np.tile(y_t, (time_len, 1)),
                         np.tile(z_t, (time_len, 1)),
                         weightsDM[0], weightsDM[1], weightsDM[2], 
                         weightsDM[3], weightsDM[4], weightsDM[5], 
                         weightsDM[6], weightsDM[7], weightsDM[8])
        for k in range(len(weightsDM)):
            cum_grads[k] += tmp[k]
        cum_error += tmp[len(weightsDM)]
    print "Epoch: %d [Cost: %.3f]" % (i, 200 * cum_error/batch_size)
    # Batch weight updates
    for k in range(len(weightsDM)):
        updates[k] = momentum * updates[k] + (learning_rate[k] * cum_grads[k]/batch_size)
    for k in range(len(weightsDM)):
        weightsDM[k] -= updates[k]
        learning_rate[k] *= learning_rate_updates[k]

np.savez("DM_eq.npz", 
    W0=weightsDM[0], W1=weightsDM[1], W2=weightsDM[2],
    W3=weightsDM[3], W4=weightsDM[4], W5=weightsDM[5],
    W6=weightsDM[6], W7=weightsDM[7], W8=weightsDM[8])

# Guidelines
# l: LHS
# c: Comparison Matrix
# p: Production Fire Vector
# b: LHS broadcast Matrix
# g: LHS broadcast gating Matrix
# r: RHS

# Theano Inputs/Targets
l = TT.matrix()
c_target = TT.matrix()
p_target = TT.matrix()
r_target = TT.matrix()
train_lhs = TT.scalar()
train_rhs = TT.scalar()
# Initial Sequence Values
c0 = theano.shared(np.zeros(num_slots**2))
p0 = theano.shared(np.zeros(num_rules))
r0 = theano.shared(np.zeros(num_slots))
# Theano Weights and Biases
W_lc = TT.matrix()
W_lp = TT.matrix()
W_lb = TT.matrix()
W_cc = TT.matrix()
W_cp = TT.matrix()
W_pc = TT.matrix()
W_pp = TT.matrix()
W_pg = TT.matrix()
W_pr = TT.matrix()
W_bgr = TT.matrix()
b_c = TT.vector()
b_p = TT.vector()
W_list = [W_lc, W_lp, W_lb, W_cc, W_cp, W_pc, W_pp, W_pg, W_pr, W_bgr, b_c, b_p]

# Step function to compute the production rule network
def step(l_t, c_tm1, p_tm1, r_tm1, 
    W_lc, W_lp, W_lb, W_cc, W_cp, W_pc, W_pp, W_pg, W_pr, W_bgr, b_c, b_p):
    # Compute the comparison matrix at time t
    c_t = TT.nnet.sigmoid(TT.dot(l_t, W_lc) + TT.dot(c_tm1, W_cc) + TT.dot(p_tm1, W_pc) + b_c)
    # Compute the production rule fire vector at time t
    p_t = TT.nnet.sigmoid(TT.dot(l_t, W_lp) + TT.dot(c_t, W_cp) + TT.dot(p_tm1, W_pp) + b_p)
    # Compute the RHS final product
    r_t = TT.dot(TT.dot(l_t, W_lb) * TT.nnet.sigmoid(TT.dot(p_t, W_pg)), W_bgr) + TT.dot(p_t, W_pr)
    return c_t, p_t, r_t

# Scan using the function the link the recurrent network
[c, p, r], _ = theano.scan(step, sequences=[l], outputs_info=[c0, p0, r0], non_sequences=W_list)

# Error function
comp_error = ((c-c_target)**2).mean().mean()
fire_error = ((p-p_target)**2).mean().mean()
recon_error = ((r[-1]-r_target[-1])**2).mean()
error = train_lhs * (comp_error + fire_error) + train_rhs * recon_error

# Get the list of gradients
[G_lc, G_lp, G_lb, G_cc, G_cp, G_pc, G_pp, G_pg, G_pr, G_bgr, G_c, G_p] = TT.grad(error, W_list)
G_list = [G_lc, G_lp, G_lb, G_cc, G_cp, G_pc, G_pp, G_pg, G_pr, G_bgr, G_c, G_p]
# Training Function
train_fn = theano.function([l, c_target, p_target, r_target, train_lhs, train_rhs,
                            W_lc, W_lp, W_lb, W_cc, W_cp, W_pc, W_pp, W_pg, W_pr, W_bgr, b_c, b_p],
                           [G_lc, G_lp, G_lb, G_cc, G_cp, G_pc, G_pp, G_pg, G_pr, G_bgr, G_c, G_p, error])
test_fn = theano.function([l, W_lc, W_lp, W_lb, W_cc, W_cp, W_pc, W_pp, W_pg, W_pr, W_bgr, b_c, b_p],
                           [p, r])

# Learning rate         [W_lc, W_lp, W_lb, W_cc, W_cp, W_pc, W_pp,   W_pg, W_pr, W_bgr, b_c, b_p]
learning_rate =         [ 100,    3,    3,  100,    3,  100,    3,1000000,    3,    3,  100,   3]
learning_rate_updates = [.995, .999, .999, .999, .999, .995, .999,   .999, .999, .999, .995,.999]

# Other important variables
batch_size = 10
momentum = 0.90
time_len = 2
epoch = 6000
epoch_rhs = epoch / 2
epoch_both = epoch

# List of numpy weights
weights = [
np.random.uniform(size=(num_slots, num_slots**2), low=-.01, high=.01),    #W_lc
np.random.uniform(size=(num_slots, num_rules), low=-.01, high=.01),       #W_lp
np.random.uniform(size=(num_slots, num_slots**2), low=-.01, high=.01),    #W_lb
np.random.uniform(size=(num_slots**2, num_slots**2), low=-.01, high=.01), #W_cc
np.random.uniform(size=(num_slots**2, num_rules), low=-.01, high=.01),    #W_cp
np.random.uniform(size=(num_rules, num_slots**2), low=-.01, high=.01),    #W_pc
np.random.uniform(size=(num_rules, num_rules), low=-.01, high=.01),       #W_pp
np.random.uniform(size=(num_rules, num_slots**2), low=-.01, high=.01),    #W_pg
np.random.uniform(size=(num_rules, num_slots), low=-.01, high=.01),       #W_pr
np.random.uniform(size=(num_slots**2, num_slots), low=-.01, high=.01),    #W_bgr
np.random.uniform(size=(num_slots**2), low=-.01, high=.01),               #b_c
np.random.uniform(size=(num_rules), low=-.01, high=.01)                   #b_p
]

for i in range(epoch):
    # Set up the batch cumulative gradients
    cum_grads = [
        np.zeros((num_slots, num_slots**2)),    #W_lc
        np.zeros((num_slots, num_rules)),       #W_lp
        np.zeros((num_slots, num_slots**2)),    #W_lb
        np.zeros((num_slots**2, num_slots**2)), #W_cc
        np.zeros((num_slots**2, num_rules)),    #W_cp
        np.zeros((num_rules, num_slots**2)),    #W_pc
        np.zeros((num_rules, num_rules)),       #W_pp
        np.zeros((num_rules, num_slots**2)),    #W_pg
        np.zeros((num_rules, num_slots)),       #W_pr
        np.zeros((num_slots**2, num_slots)),    #W_bgr
        np.zeros((num_slots**2)),               #b_c
        np.zeros((num_rules))                   #b_p
    ]
    updates = copy.deepcopy(cum_grads)
    cum_error = 0

    if (i >= epoch_both):
        train_lhs, train_rhs = 1, 1
    elif (i >= epoch_rhs):
        train_lhs, train_rhs = 0, 1
    else: 
        train_lhs, train_rhs = 1, 0

    # Run a batch
    for j in range(batch_size):
        l_t, c_t, p_t, r_t = generateData(global_var_map, prod_list, i * batch_size + j)
        l_t = nil_encode(l_t, number_repeats)
        c_t = encode(c_t, number_repeats**2)
        r_t = nil_encode(r_t, number_repeats)
        tmp = train_fn(np.tile(l_t, (time_len, 1)),
                       np.tile(c_t, (time_len, 1)),
                       np.tile(p_t, (time_len, 1)),
                       np.tile(r_t, (time_len, 1)),
                       train_lhs, train_rhs,
                       weights[0], weights[1], weights[2], weights[3],
                       weights[4], weights[5], weights[6], weights[7],
                       weights[8], weights[9], weights[10], weights[11])
        for k in range(len(weights)):
            cum_grads[k] += tmp[k]
        # print printProductionFireVector(p_t, prod_list), np.around(tmp[len(weights)] * 200, decimals=5)
        cum_error += tmp[len(weights)]
    print "Epoch: %d [Cost: %.3f]" % (i, 200 * cum_error/batch_size)

    # Batch weight updates
    for k in range(len(weights)):
        updates[k] = momentum * updates[k] + (learning_rate[k] * cum_grads[k]/batch_size)

    if (i >= epoch_both):
        update_list = [0,1,2,3,4,5,6,7,8,9,10,11]
    elif (i >= epoch_rhs):
        update_list = [2,7,8,9]
    else:
        update_list = [0,1,3,4,5,6,10,11]
    for k in update_list:
        weights[k] -= updates[k]
        learning_rate[k] *= learning_rate_updates[k]

np.savez("PR_eq.npz", 
    W0=weights[0], W1=weights[1], W2=weights[2],
    W3=weights[3], W4=weights[4], W5=weights[5],
    W6=weights[6], W7=weights[7], W8=weights[8],
    W9=weights[9], W10=weights[10], W11=weights[11])

total_error = 0
total_timeout = 0
total_5 = 0

for a in range(11):
  for b in range(11):
    subl = [1.0 * a / 10, 1.0 * b / 10, -1, -1, -1, -1]
    if (a + b > 10):
        continue
    print "%3d + %3d = " % (a, b),
    total_5 += abs(a+b-5)
    i = 0
    while (1):
        l_t = np.asarray(subl)
        l_t = nil_encode(l_t, number_repeats)
        p, r = test_fn(np.tile(l_t, (time_len, 1)),
                       weights[0], weights[1], weights[2], weights[3],
                       weights[4], weights[5], weights[6], weights[7],
                       weights[8], weights[9], weights[10], weights[11])
        r = nil_decode(r[-1], number_repeats)
        
        """
        print "(P"
        for k in range(time_len):
            printProductionFireVector(p.tolist()[k], prod_list)
            print np.around(p.tolist()[k], decimals=3)
        print
        printProductionRuleVector(subl, global_var_map)
        print "==>"
        printProductionRuleVector(r.tolist(), global_var_map)
        print ")"
        """

        if (p.tolist()[-1][1] == max(p.tolist()[-1]) or i >= b * 2 + 2):
            if (i >= b * 2 + 2):
                print "%5d (%5d) [%5s]" % ((np.around(r[3]*10, decimals=0)), a+b, "TIMEOUT")
                total_timeout += 1
                total_error += abs(a+b-np.around(r[3]*10, decimals=0))
                break
            print "%5d (%5d) [Error: %5d]" % (np.around(r[3]*10, decimals=0), a+b, abs(a+b-(np.around(r[3]*10, decimals=0))))
            total_error += abs(a+b-np.around(r[3]*10, decimals=0))
            break

        subl = r.tolist()
        subl[5] = subl[4] + 0.1
        '''
        #print
        #print "QUERY RETRIEVAL: ", np.around(r[4]*10, decimals=0), np.around(r[5]*10, decimals=0)
        x_t = nil_encode(r, number_repeats)
        y, z = testDM_fn(np.tile(x_t, (time_len, 1)),
                         weightsDM[0], weightsDM[1], weightsDM[2], 
                         weightsDM[3], weightsDM[4], weightsDM[5], 
                         weightsDM[6], weightsDM[7], weightsDM[8])
        z = nil_decode(z[-1], number_repeats)
        z = np.around(z, decimals=1)

        #print "GOT RETRIEVAL: ", np.around(z[4]*10, decimals=0), np.around(z[5]*10, decimals=0)
        subl = z.tolist()
        '''
        i += 1

print "Total Error: %d Total TIMEOUT: %d Total 5: %d" % (total_error, total_timeout, total_5)

