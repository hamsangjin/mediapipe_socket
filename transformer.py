import re
import json

input_str = """
landmark {
  x: 0.021526403725147247
  y: -0.598235547542572
  z: -0.34839940071105957
  visibility: 0.9998868703842163
}
landmark {
  x: 0.028281109407544136
  y: -0.6313484311103821
  z: -0.3350728154182434
  visibility: 0.9998244047164917
}
"""

pattern = r"landmark {\s+x: ([+-]?\d+\.\d+)\s+y: ([+-]?\d+\.\d+)\s+z: ([+-]?\d+\.\d+)"
matches = re.findall(pattern, input_str)

output_data = {"left": [], "right": [], "pose": []}

for match in matches:
    x, y, z = map(float, match)
    output_data["left"].append({"x": round(x, 3), "y": round(y, 3), "z": round(z, 3)})
    output_data["right"].append({"x": round(x, 3), "y": round(y, 3), "z": round(z, 3)})
    output_data["pose"].append({"x": round(x, 3), "y": round(y, 3), "z": round(z, 3)})

output_json = json.dumps(output_data, indent=2, ensure_ascii=False)

print(output_json)
