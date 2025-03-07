document.addEventListener("DOMContentLoaded", function () {

  async function fetchProblem() {
    const difficulty = document.querySelector("input[name='difficulty']:checked");
    const handle =  document.getElementById("handle").value.trim(); // Get input
    const tag = encodeURIComponent(document.getElementById("problemTag").value);


    const apiUrl = `http://172.20.10.10:8000/${handle}?difficulty=${difficulty.value}&tag=${tag}`;

    document.getElementById("message").textContent = "Loading...";
    try {
      const response = await fetch(apiUrl, {
        method: "GET",
        headers: { "Accept": "application/json" }
      });

      if (!response.ok) throw new Error("Failed to fetch data");

      const data = await response.json();
      document.getElementById("message").textContent = "";

      const result1 = data.problems_list[0];
      var link1 = document.createElement("a");
      link1.href = result1.url;
      link1.textContent = result1.name;
      link1.target = "_blank"; // Open in a new tab
      document.getElementById("problem1").appendChild(link1); 
      var textNode1 = document.createTextNode(` (${result1.rating})`);
      document.getElementById("problem1").appendChild(textNode1); 

      const result2 = data.problems_list[1];
      var link2 = document.createElement("a");
      link2.href = result2.url;
      link2.textContent = result2.name;
      link2.target = "_blank"; // Open in a new tab
      document.getElementById("problem2").appendChild(link2); 
      var textNode2 = document.createTextNode(` (${result2.rating})`);
      document.getElementById("problem2").appendChild(textNode2); 

      const result3 = data.problems_list[2];
      var link3 = document.createElement("a");
      link3.href = result3.url;
      link3.textContent = result3.name;
      link3.target = "_blank"; // Open in a new tab
      document.getElementById("problem3").appendChild(link3); 
      var textNode3 = document.createTextNode(` (${result3.rating})`);
      document.getElementById("problem3").appendChild(textNode3); 

      const result4 = data.problems_list[3];
      var link4 = document.createElement("a");
      link4.href = result4.url;
      link4.textContent = result4.name;
      link4.target = "_blank"; // Open in a new tab
      document.getElementById("problem4").appendChild(link4); 
      var textNode4 = document.createTextNode(` (${result4.rating})`);
      document.getElementById("problem4").appendChild(textNode4); 

      const result5 = data.problems_list[4];
      var link5 = document.createElement("a");
      link5.href = result5.url;
      link5.textContent = result5.name;
      link5.target = "_blank"; // Open in a new tab
      document.getElementById("problem5").appendChild(link5); 
      var textNode5 = document.createTextNode(` (${result5.rating})`);
      document.getElementById("problem5").appendChild(textNode5); 
    } catch (error) {
      //document.getElementById("message").innerText = "Error: " + error.message;
    }

   /*
    document.getElementById("message").textContent = "Difficulty: " + difficulty.value; // Update screen
    document.getElementById("problem1").textContent = "Handle: " + handle; // Update screen
    document.getElementById("problem2").textContent = "tag: " + tag;
    */
  }

  const button = document.getElementById("sendRequest");
  button.addEventListener("click", fetchProblem);
  document.addEventListener("keydown", function (event) {
    if (event.key === "Enter") {
      event.preventDefault(); // Prevent default form submission (if any)
      button.click(); // Simulate button click
    }
  });
});
