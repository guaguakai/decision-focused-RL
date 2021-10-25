import pandas as pd
import numpy as np
import random


def generate_T_matrices_random(
    NUM_T_MATRICES=100
):
    # Generate random matrices
    patients_matrices = np.random.rand(NUM_T_MATRICES, 2, 2)
    patients_matrices += 0.2 * np.eye(2)  # make states `sticky', i.e., patients more likely to stay in the state they're in
    patients_matrices /= patients_matrices.sum(axis=-1)[:, :, np.newaxis]  # normalise to get probabilities

    return patients_matrices


def generate_T_matrices_from_data(
    DATA_PATH,
    MIN_SEQ_LEN,
):
    # LOAD DATA
    df = pd.read_csv(DATA_PATH, parse_dates=['EnrollmentDate'])

    # PROCESS DATA
    df = df[df['EnrollmentDate'] < pd.Timestamp('8/1/17')]

    #  Map `codes' to states
    #   `Codes' in the dataset
    #   1: end date (no distinction for whether a call was made on this date)
    #   4: confirmed taken dose via unshared number
    #   5: confirmed taken dose via shared number
    #   6: missed dose
    #   8: enrollment date (no distinction for whether a call was made on this date)
    #   9: manual dose (didn't receive a call, but some provider marked the patient as having
    #      taken the dose)
    code_dict = {
        '4': 1,
        '5': 1,
        '6': 0,
        '9': 1,

        '1': 1,
        '8': 1
    }

    def convert_to_binary_sequence(sequence):
        return [code_dict[i] for i in sequence]

    df['AdherenceString'] = df['AdherenceString'].astype(str)
    df['AdherenceSequence'] = df['AdherenceString'].apply(convert_to_binary_sequence)
    sequences = df['AdherenceSequence'].values

    # Convert adherence values to T matrices
    patients_matrices = []
    for sequence in sequences:
        patient_T_matrix = np.ones(tuple([2] * (2)))
        if len(sequence) > MIN_SEQ_LEN:
            j = 0
            while j + 1 < len(sequence):
                curr_state = sequence[j]
                next_state = sequence[j + 1]
                patient_T_matrix[curr_state, next_state] += 1
                j += 1

            totals = patient_T_matrix.sum(axis=-1)
            if (np.any(totals == 0)):
                continue  # ensure that there has been at least one observed transition per state
            patient_T_matrix /= totals[:, np.newaxis]
            patients_matrices.append(patient_T_matrix)

    return patients_matrices


def verify_T_matrix(T, EPSILON):
    valid = True
    valid &= T[0, 0, 1] - T[0, 1, 1] <= EPSILON  # non-oscillate condition
    valid &= T[1, 0, 1] - T[1, 1, 1] <= EPSILON  # must be true for active as well
    valid &= T[0, 1, 1] <= T[1, 1, 1]  # action has positive "maintenance" value
    valid &= T[1, 0, 0] <= T[0, 0, 0]  # action has non-negative "influence" value
    return valid


def generate_action_effects(
    T_matrices,
    NUM_PATIENTS,
    EFFECT_SIZE,
    EPSILON,
):
    """
    Generates a matrix indexed as: T[patient_number][action][current_state][next_state]
    action=0 denotes passive action, a=1 is active action
    State 0 denotes NA and state 1 denotes A
    """
    # For each patient
    patient_models = []
    while len(patient_models) < NUM_PATIENTS:
        # Sample a transition matrix
        T = random.choice(T_matrices)

        # ADD ACTION EFFECCTS
        shift = EFFECT_SIZE

        # Define Action Effects
        #   Patient responds well to call
        benefit_act_00 = np.random.uniform(low=0., high=shift)  # will subtract from prob of staying 0,0
        benefit_act_11 = benefit_act_00 + np.random.uniform(low=0., high=shift)  # will add to prob of staying 1,1
        #   add benefit_act_00 to benefit_act_11 to guarantee the p11>p01 condition

        #   Patient does well on their own, low penalty for not calling
        penalty_pass_11 = np.random.uniform(low=0., high=shift)  # will sub from prob of staying 1,1
        penalty_pass_00 = penalty_pass_11 + np.random.uniform(low=0., high=shift)  # will add to prob of staying 0,0

        # Apply Action Effects
        # Note: This is slightly different from Jackson and Aditya's implementation
        T_pass = np.copy(T)
        T_act = np.copy(T)

        T_act[0, 0] = np.clip(T_act[0, 0] - benefit_act_00, EPSILON, 1 - EPSILON)
        T_act[1, 1] = np.clip(T_act[1, 1] + benefit_act_11, EPSILON, 1 - EPSILON)

        T_pass[0, 0] = np.clip(T_pass[0, 0] + penalty_pass_00, EPSILON, 1 - EPSILON)
        T_pass[1, 1] = np.clip(T_pass[1, 1] - penalty_pass_11, EPSILON, 1 - EPSILON)

        # Re-Normalise
        T_pass[0, 1] = 1 - T_pass[0, 0]
        T_pass[1, 0] = 1 - T_pass[1, 1]

        T_act[0, 1] = 1 - T_act[0, 0]
        T_act[1, 0] = 1 - T_act[1, 1]

        # Verify and Assign
        patient_model = np.array([T_pass, T_act])
        if not verify_T_matrix(patient_model, EPSILON):
            print("Ignoring possibly problematic T matrices:\n", patient_model)
            continue
        patient_models.append(patient_model)

    patient_models = np.array(patient_models)
    return patient_models
