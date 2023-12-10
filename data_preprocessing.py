import openfoamparser_mai as Ofpp
import os


def parse_and_preprocess_openfoam_data(foambase_directory):
  """
  Parses OpenFOAM data and preprocesses it for the DNN model.

  Args:
    foambase_directory: Base directory of the OpenFOAM simulation.

  Returns:
    data_list: List of tuples containing input and target data.
  """
  data_list = []
  # Define relevant fields for parsing
  fields = ["U", "alpha.water", "p"]

  # Iterate through time directories
  for time_dir in os.listdir(foambase_directory):
    if not os.path.isdir(os.path.join(foambase_directory, time_dir)):
      continue

    # Extract data for each field
    field_data = {field: Ofpp.parse_internal_field(os.path.join(foambase_directory, time_dir, field))
                  for field in fields}

    # Calculate additional features (e.g., velocity magnitude)
    velocity_magnitude = np.linalg.norm(field_data["U"], axis=1)

    # Combine features with additional parameters
    data_point = (field_data["alpha.water"], velocity_magnitude, field_data["p"])

    # Extract target data (e.g., lift, drag, moment)
    # Replace with your specific target data parsing method
    lift, drag, moment = ...

    # Create data tuple
    data_list.append((data_point, (lift, drag, moment)))

  return data_list

