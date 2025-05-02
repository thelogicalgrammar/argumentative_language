<template>
  <Screen v-bind="$props">

    <slot name="stimulus"></slot>

    <template>
      <Record
        :data="{
          question,
          options,
          text
        }"
      />

      <p v-html="question"></p>

      <ButtonCompletionInput
        :text="text"
        :options="options"
        @updateResponse="updateResponseFunc"
        @updateResponses="updateResponsesFunc"
      />

      <button
        v-if="
          nextButtonActive
        "
        @click="$magpie.saveAndNextScreen()"
      >
        Next
      </button>

    </template>
  </Screen>

</template>

<script>
import ButtonCompletionInput from '@/ButtonCompletionInput.vue';

export default {
  name: 'SelectionScreen',
  components: {
	ButtonCompletionInput
  },
  data () {
  	return {
		nextButtonActive: false
	}
  },
  methods: {
	updateResponseFunc (fullAnswer){
		$magpie.measurements.completed_text = fullAnswer;
	},
	updateResponsesFunc (answers){
		$magpie.measurements.responses = answers;
		this.nextButtonActive = $magpie.measurements.responses &&
			$magpie.measurements.responses.filter(Boolean).length === this.options.length;
	}
  },
  props: {
    question: {
      type: String,
      required: true
    },
    text: {
      type: String,
      required: true
    },
    options: {
      type: Array,
      required: true
    },
	progress: {
	  type: Number,
	  required: false
    }
  }
};
</script>
