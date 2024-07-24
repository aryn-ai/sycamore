import { render as rtlRender, RenderOptions } from "@testing-library/react";
import { MantineProvider } from "@mantine/core";
import { theme } from "../pages/HomePage";
import { ComponentType, ReactElement } from "react";

export const AllTheProviders = ({
  children,
}: {
  children: React.ReactNode;
}) => {
  // console.log("children" + JSON.stringify(children));
  // console.log("MantineProvider" + JSON.stringify(MantineProvider));
  const comp = <MantineProvider>{children}</MantineProvider>;
  // console.log("comp" + JSON.stringify(comp));
  return comp;
};
export const render = async (
  ui: React.ReactNode,
  options?: Omit<RenderOptions, "wrapper">,
) => {
  const result = rtlRender(ui, {
    wrapper: AllTheProviders,
    ...options,
  });

  return result;
};

// Same implementation as above
// export const render = (children: ReactElement) => {
//   return rtlRender(children, {
//     wrapper: ({ children }: { children: React.ReactNode }) => (
//       <MantineProvider theme={theme}>{children}</MantineProvider>
//     ),
//   });
// };

export * from "@testing-library/react";
