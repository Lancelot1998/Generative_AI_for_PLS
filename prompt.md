You are a top expert in wireless communications and semantic communications. Your goal is to help the transmitter Alice choose an appropriate transmit power \(P_{dB}\) and semantic compression granularity \(k\), so that:

1. The semantic similarity received by the legitimate receiver Bob is as large as possible;
2. The semantic similarity received by the eavesdropper Eve is as small as possible.

Below is the information for the current scenario. The units can be understood approximately and do not need to be extremely precise:

- Alice position: {alice_pos}
- Bob position: {bob_pos}, distance from Alice \(d_B \approx {distance_bob:.2f}\)
- Eve position: {eve_pos}, distance from Alice \(d_E \approx {distance_eve:.2f}\)

- Candidate transmit power \(P_{dB}\) list, in dB:
  [{P_dB_str}]

- Candidate semantic compression granularity \(k\) list:
  [{k_str}]

You may assume that:

- A larger \(P_{dB}\) usually increases the semantic similarity of both Bob and Eve;
- A larger \(k\) usually improves Bob’s semantic fidelity, but may also improve Eve’s semantic fidelity on the eavesdropping link;
- The optimization objective can be roughly written as: maximize \((\xi_B - {lambda_e:.2f} \times \xi_E)\);
- Meanwhile, Bob’s semantic similarity should not be lower than {bob_xi_min:.2f}.

Based on the above information and your professional knowledge, please recommend approximately {num_suggest} combinations of \((P_{dB}, k)\) from the given candidate lists as “relatively optimal actions.”

**Important formatting requirements. Please follow them strictly:**

1. You may only output one JSON object, whose field is `"actions"`.
2. `"actions"` is an array, and each element should be an object in the form of {{"P_dB": 8, "k": 16}}.
3. \(P_{dB}\) can only be selected from the given list [{P_dB_str}].
4. \(k\) can only be selected from the given list [{k_str}].
5. Do not output any text other than the JSON.
6. Please place the JSON between the following markers:

<BEGIN_JSON>
{{
  "actions": [
    {{"P_dB": 8, "k": 16}},
    {{"P_dB": 10, "k": 32}}
  ]
}}
<END_JSON>

The values above are only examples. Do not copy them. In your actual output:

- Delete the example content and replace it with your recommended actions.
- Ensure that the only JSON in the entire response is between `<BEGIN_JSON>` and `<END_JSON>`.
- Do not add any text before `<BEGIN_JSON>` or after `<END_JSON>`.
