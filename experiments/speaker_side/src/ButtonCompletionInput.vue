// Component for the screen with the stimulus and utterance selection

<template>
  <div class="completion">
	<div style='display:inline' v-for="(slice, i) in slices" :key="i" >
	<div style='margin:15px; display:inline'> {{ slice }} </div>
	<StackedButtons
	  v-if="i !== slices.length - 1"
	  @updateButton="updateAnswers"
	  :opts="options[i]"
	  :index="i"
	/>
	</div>
  </div>
</template>

<script>
import StackedButtons from '@/StackedButtons.vue';


export default {
  name: 'ButtonCompletionInput',
  props: {
    text: {
      type: String,
      required: true
    },
    options: {
      type: Array,
      required: true
    }
  },
  components: {
  	StackedButtons
  },	
  data() {
    return {
      answers: []
    };
  },
  methods: {
	updateAnswers(name, index){
	  // update this.answers
	  this.answers[index] = name;
	  var answers = this.fullAnswer();
	  // Emit event to update full response
	  this.$emit('updateResponse', this.fullAnswer());
	  //Emit event to update Array of choices
	  this.$emit('updateResponses', this.answers);
	},
	fullAnswer() {
	  // whenever answers change, 
	  // text is updated automatically
	  const answers = this.answers.slice();
	  return this.text
		.split('%s')
		.map((s) => s + (answers.shift() || ''))
		.join('');
	},
  },
  computed: {
    slices() {
      return this.text.split('%s');
    },
  }
};
</script>

<style scoped>
	textarea {
	  border: 2px solid #5187ba;
	  border-radius: 10px;
	  display: block;
	  font-size: 16px;
	  margin: 0 auto;
	  outline: none;
	  padding: 10px 20px;
	}

	div > * {
		vertical-align: middle;
		line-height: normal;
	}
</style>
