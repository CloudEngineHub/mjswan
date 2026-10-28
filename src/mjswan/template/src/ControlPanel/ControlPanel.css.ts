import { style } from "@vanilla-extract/css";

/** The panel's logo opens the About dialog; the hover shows it is a button. */
export const aboutButton = style({
  padding: "0.2em",
  borderRadius: "var(--mantine-radius-sm)",
  transition: "background-color 100ms",
  ":hover": {
    backgroundColor: "var(--mantine-color-gray-light-hover)",
  },
});
