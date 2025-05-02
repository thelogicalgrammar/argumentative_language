<template>
	<div v-html="tableHTML" />
</template>

<script>
import _ from 'lodash';

function tableGeneratorSituation(trialdata) {
    function createRow(right, total){
		// Return an Array with 0s and 1s
		return Array(right).fill(1).concat(Array(total-right).fill(0))
    };

    function getCell(result) {
		if (result == 1) {
			return "<td bgcolor='#59b370'><i style=color:#073813>&#10004</i></td>";
		} else {
			return "<td bgcolor='#f26049'><i style=color:#e0beba>&#10008</i></td>";
		}
    };

    function getRow(tableName, rowNumbers){
		// Create a row by adding an initial name followed by a cell
		// for each answer.
		return '<tr><th>' + tableName + '</th>' + rowNumbers.map((n)=> getCell(n)) + '</tr>';
    };

	// Create a matrix with 0s and 1s
    var matrix = trialdata.studentsArray.map(
		(n) => createRow(n, trialdata.nQuestions)
	);

	var tableNames = trialdata.names;
    var result = "<table border=1>" 
		+ matrix.map((rownumbers, index) => getRow(tableNames[index], rownumbers)) 
		+ '</table>';
	// replace statement eliminates some annoying commas 
	// that are automatically inserted
    return result.replace(/,/g, "");
}

export default {
	methods: {
		// Make function to render table accessible to template above
		tableGeneratorSituation
	},
	props: ['trialData'],
	computed: {
		// Expose lodash to template code
		_() { return _; },
		tableHTML(){return tableGeneratorSituation(this.trialData)}
	}
};

</script>

<style>
table {
	margin: 0 auto; 
}
</style>
