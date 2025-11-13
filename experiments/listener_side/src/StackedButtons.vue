// Code for the group of vertical buttons, one of which needs to be selected

<template>
	<div class="btn-group" style="display:inline-block">
		<button 
			v-for="(opt, i) in opts" 
			:key='opt' 
			v-text='opt'
			@click="updateClick(opt, i)"
			:class="{ pressed: active === i }"
		>
		</button>
	</div>
</template>

<script>

// index of active button
var active;

export default {
  name: 'StackedButtons',
  props: [
    // opts is the set of options to choose from
	// (one for each button)
  	'opts',
	// index is the index of the button group
	'index'
  ],
  data() {
    return {
		active,
    }
  },
  methods: {
  	updateClick(name, buttonIndex){
		// Logic for buttons press
		// change property of selected button
		// and de-select all other buttons
		this.active = buttonIndex;
		this.$emit('updateButton', name, this.index);
	}
  },
};

</script>

<style scoped>
	.btn-group button {
	  /* background-color: #808080; /1* Green background *1/ */
	  border: 1px solid grey; /* Green border */
	  color: white; /* White text */
	  padding: 1px 4px; /* Some padding */
	  cursor: pointer; /* Pointer/hand icon */
	  width: 100%; /* Set a width if needed */
	  display: block; /* Make the buttons appear below each other */
	  margin: 10px 10px;
	}

	.btn-group button:not(:last-child) {
	  border-bottom: none; /* Prevent double borders */
	}

	/* Add a background color on hover */
	.btn-group button:hover {
	  background-color: blue;
	}

	.pressed {
	  background-color: blue;
	}
</style>
